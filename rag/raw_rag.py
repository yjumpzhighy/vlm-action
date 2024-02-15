import torch
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
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredMarkdownLoader, UnstructuredURLLoader
from langchain.chains import LLMChain, SimpleSequentialChain, RetrievalQA, ConversationalRetrievalChain
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
import warnings
warnings.filterwarnings('ignore')

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

result = llm(prompt.format(text="How about the company AutoX?"))
print(result)


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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
texts_chunks = text_splitter.split_documents(documents)
# for chunk in texts_chunks:
#     print(len(chunk.page_content))


# construct vector storage
vector_db = Chroma.from_documents(texts_chunks, 
                                  HuggingFaceEmbeddings(
                                    model_name="thenlper/gte-large",
                                    model_kwargs={"device": "cuda"},
                                    encode_kwargs={"normalize_embeddings": True},
                                  ), 
                                  persist_directory="db"
)
#print(vector_db.similarity_search("Auto CEO"))

template = """
[INST] <>
Act as a investment analyst who search latest news.
Use the following information to answer the question at the end.
<>

{context}

{question} [/INST]
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)

result_ = qa_chain("How about the company AutoX?")
print(result_["result"].strip())


