
# RAG: Retrieval,Augmented,Generate
RAG is technique aiming at optimize llm output, forcing the model to retrieval info from provided 
dataset as reference to generate final query response. Generally include 4 steps: 
1) Embedding: use pre-trained embedding model to embed documents, transfer text into number vectors. 
2) Store: put embedding vectors into vector storage, like ChromaDB or FAISS. Those data structures 
          are specialized for fast indexing, storage, similarity search, etc. 
3) Query: with your query, the embedding model would embed the query first, and then compare 
          and search the most similar text with vector cosine metrics. 

        #compute cosine similarity example
        q = [a1, a2, a3]
        s = [s1, s2, s3, s4, s5]
        #padding the shorter vector to same dimension
        q = [a1, a2, a3, 0, 0]
        similary = (a1*s1 + a2*s2 + a3*s3) / (sqrt(a1*a1+a2*a2+a3*a3) + sqrt(s1*s1+s2*s2+s3*s3+s4*s4+s5*s5))
        distance = 1 - similarity

4) Prompt: together with your query, the most similar context in the vector storage will be combined 
           into prompt, feedinto the llm for response. 

Query results' quality from retriever is critical to RAG performance, the most common evalution
metrics are Hit Rate and Mean Reciprocal Rank.
HR: It calculates the fraction of queries where the correct answer is found within the top-k retrieved documents. 
MRR: It computes a score indicating how high up in the list the first correctly retrieved document is. 
     For each query, MRR looks at the rank of the highest-placed relevant document. Specifically, it's the average 
     of the reciprocals of these ranks across all the queries. So, if the first relevant document is the top result, 
     the reciprocal rank is 1; if it's second, the reciprocal rank is 1/2, and so on.


## 1. raw rag
raw rag usage is straightforward and simply following classic rag steps. 
we use llama2 to ask about an autonomous driving tech startup company autox as example. 

        python rag/raw_rag.py

When direct ask llama2 about autox, it responds autox is an EV company while giving sufficent specious and 
groundless details, wrong answers!!! 

With RAG, we provided some autox news and inject into the prompt, the llama2 responds autox is an autonomous 
company with detailed facts, from the web news provided, right answers!!! 


## 2. self-rag 
raw rag is able to boost llm output quanlity (like hallucination) by augument input content. However, it 
has following issues:
1) the provided documents may be misleading and low-quanlity contents, resulting in inaccurte LLM answers.
2) selected top k textnodes may not cover all required information.
3) cosine similarity doesn't guarantee strong related context.
4) unsatisfactory performance on complex task question.

In "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (by Akari Asai, Zeqiu Wu, etc),
"retrieval on demand" and "reflection tokens" strategies are used to optimize above issues.


## 3. advanced rag
### 3.1 child-parent recursive retriever
Instead of large-size chunk (parent chunk), use small chunk (child chunk) for retrieval. Then use its's 
corresponding large-size chunk (parent chunk) for sythetizer. 

In another word, documents text are indexed into retrieval and sythesis stores seperately, while keep their
nodes' index relations. In retrieval store, documents text divided into small chunks for better relevant sentences
locating. In systhesis store, documents text divided into large chunks to provide sufficient LLM input context.

        python rag/advanced_rag_child_parent_retrieval.py

### 3.2 reranker
embedding similarity is relatively intuitive but not always accurate. During experiments, the top1 relevant document 
is usually reliable, but top2 ~ top10 (if return 10 documents) documents usually not ranked by relevance.
In another, the retrieved documents by embedding search are not always "actual top". And most importantly, the ranking 
is not trustable.

Hense, introducing reranker, which use initial retrieval results as input and optimize their ranking.

        python rag/advanced_rag_rerank.py

Note: common used rerankers like CohereRerank and bge-rerank-large shows satisfactory improvement in most QA tasks.

### 3.3 multi-modality 
TODO

### 3.4 contextual compression
When we divide document blocks, we often do not know the user's query. This means that the most relevant information to 
the query may be hidden in a document containing a large amount of irrelevant text. Inputting this to LLM may result in 
more expensive LLM calls and poorer responses.

"Compression" refers to both compressing contents of document and filtering out documents. Contextual compression would
compress initial retrieval results using the context of the given query, and return only relevant information. 
Generally include 3 steps: 
1) contextual compression retriever passes queries to the base retriever and get retrieval results.
2) takes the initial documents and passes them through the document compressor, which takes a list of documents and shortens 
   it by reducing the contents of documents or dropping documents altogether.
3) return compressed content and your query combined into prompt, feedinto the llm for response. 

        python rag/advanced_rag_compression.py

        # compressor pipeline uses compressor(pretrained llm model) to extract key content, relevant_filter to filter
        # out embeddings with low similarity, redundant_filter to filter out duplicate output documents.

Note: in experiment, notice that adding compressor would add additional noise to final answer, requires carefully finetuning.


### 3.5 middle augment
llm generally shows "lost in the middle" phenomenon, as LLMs are more  likely to recall information positioned 
at the start or end of an input, with a tendency to overlook content in the middle.

"LIM" also exists in rag. for example, if retrieved document with answer positioned at start or end of the returned documents 
list, the final output answer would be more accurate. 

However, there is no mature solution at the moment, what we can do is refine the retrieved documents quality and rank 
"answer document"to the top, which is the key position utilizing "LIM" phenomenon. (sounds like reranking, right?)

Rather than reranking, another technique called "Lord of the Retrievers" (also known as MergerRetriever), takes a list of 
retrievers as input and merges the results of their get_relevant_documents() methods into a single list. Thus get a list of 
documents that are relevant to the query and that have been ranked by the different retrievers.
1) it can combine the results of multiple retrievers, which can help to reduce the risk of bias in the results.
2) it can rank the results of the different retrievers, which can help to ensure that the most relevant documents are 
   returned first.

        python rag/advanced_rag_merge_retrievers.py


### 3.6 query transform
TODO


### 3.7 




### 3.8 overview
Improved retrieved documents releavance: compression, child-parent, merge retrievers
Improved retrieved documents order: rerank, merge retrievers(reorder)




