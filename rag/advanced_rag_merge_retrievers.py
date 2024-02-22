import torch
import random
import numpy as np
import gradio as gr
from textwrap import fill
from IPython.display import Markdown, display
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    )
from langchain import PromptTemplate
from langchain import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.schema import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredMarkdownLoader, UnstructuredURLLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (LLMChainExtractor, 
                                                       EmbeddingsFilter, 
                                                       DocumentCompressorPipeline,
                                                       FlashrankRerank)
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.chains import LLMChain, SimpleSequentialChain, RetrievalQA, ConversationalRetrievalChain
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.document_transformers import LongContextReorder

from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
import warnings
warnings.filterwarnings('ignore')

random.seed(317)
np.random.seed(317)
torch.manual_seed(317)
torch.cuda.manual_seed(317)
torch.backends.cudnn.deterministic = True

model_name = "meta-llama/Llama-2-7b-chat-hf" #"mistralai/Mistral-7B-Instruct-v0.1"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto",
    quantization_config=quantization_config
)

generation_config = GenerationConfig.from_pretrained(model_name)
generation_config.max_new_tokens = 1024
generation_config.temperature = 0.0001
generation_config.top_p = 0.95
generation_config.do_sample = True
generation_config.repetition_penalty = 1.15
#HuggingFacePipeline can be accelerated on attention layer memory by xformer. 
llm = HuggingFacePipeline(
    pipeline=pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                return_full_text=True,
                generation_config=generation_config,
            )
)


#prompt
template = """
[INST] <>
Act as a investment analyst who search latest news.
Summarize your answer within 100 letters.
<>

{text} [/INST]
"""
prompt = PromptTemplate(
    input_variables=["text"],
    template=template,
)
print(llm(prompt.format(text="How about the company AutoX?")))



# load local data
urls = ["https://www.autox.ai/en/index.html", "https://www.autox.ai/blog/20230322.html",
        "https://www.autox.ai/blog/20220509.html", "https://www.autox.ai/blog/20220209.html",
        "https://www.autox.ai/blog/20220128.html", "https://www.autox.ai/blog/20211222.html",
        "https://www.autox.ai/blog/20211116.html", "https://www.autox.ai/blog/20210823.html",
        "https://www.autox.ai/blog/20210706.html", "https://www.autox.ai/blog/20210507.html",
        "https://www.autox.ai/blog/20210128.html", "https://www.autox.ai/blog/20201203.html",
        "https://www.autox.ai/blog/20200817.html", "https://www.autox.ai/blog/20200717.html",
        "https://www.autox.ai/blog/20200410.html", "https://www.autox.ai/blog/20200226.html",
        "https://www.autox.ai/blog/20200106.html", "https://www.autox.ai/blog/20190618.html"
]
loader = UnstructuredURLLoader(urls=urls)
documents = loader.load()

# split to chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts_chunks = text_splitter.split_documents(documents)


# construct vector storage
embedding = HuggingFaceEmbeddings(model_name="thenlper/gte-large",
                                  model_kwargs={"device": "cuda"},
                                  encode_kwargs={"normalize_embeddings": True},
)
embedding_second = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en",
                                            model_kwargs={"device":"cuda"},
                                            encode_kwargs = {'normalize_embeddings': False}
)

vector_db = Chroma.from_documents(texts_chunks, 
                                  embedding,
                                  persist_directory="db"
)
vector_db_second = Chroma.from_documents(texts_chunks, 
                                  embedding_second,
                                  persist_directory="db"
)


# Build two individual retrievers and merge into one.
retriever = vector_db.as_retriever(search_kwargs={"k":5})
retriever_second = vector_db_second.as_retriever(search_kwargs={"k":5})
lotr = MergerRetriever(retrievers=[retriever, retriever_second])


reordering = LongContextReorder()  
compressor = LLMChainExtractor.from_llm(llm=llm)
redundant_filter = EmbeddingsRedundantFilter(embeddings=embedding)
relevant_filter = EmbeddingsFilter(embeddings=embedding, similarity_threshold=0.75)
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[redundant_filter, reordering]
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor, 
    base_retriever=lotr,
    search_kwargs={"k": 5, "include_metadata": True}
)




template = """
<human>:
Context:{context}

Question:{question}

Act as an investment analyst who search latest news. Use the above Context to answer the Question.
Consider only the Context provided above to formulate response.
<bot>:

"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=compression_retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)

print(qa_chain("How about the company AutoX?")['result'])
