
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


## 1.raw rag
raw rag usage is straightforward and simply following classic rag steps. 
we use llama2 to ask about an autonomous driving tech startup company autox as example. 

        python rag/raw_rag.py

When direct ask llama2 about autox, it responds autox is an EV company while giving sufficent specious and 
groundless details, wrong answers!!! 

With RAG, we provided some autox news and inject into the prompt, the llama2 responds autox is an autonomous 
company with detailed facts, from the web news provided, right answers!!! 



## 2.self-rag 
raw rag is able to boost llm output quanlity (like hallucination) by augument input content. However, it 
has following issues:
1) the provided documents may be misleading and low-quanlity contents, resulting in inaccurte LLM answers.
2) selected top k textnodes may not cover all required information.
3) cosine similarity doesn't guarantee strong related context.
4) unsatisfactory performance on complex task question.

In "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (by Akari Asai, Zeqiu Wu, etc),
"retrieval on demand" and "reflection tokens" strategies are used to optimize above issues.

## 3.advanced rag
### 3.1 child-parent recursive retriever
Instead of large-size chunk (parent chunk), use small chunk (child chunk) for retrieval. Then use its's 
corresponding large-size chunk (parent chunk) for sythetizer. 

In another word, documents text are indexed into retrieval and sythesis stores seperately, while keep their
nodes' index relations. In retrieval store, documents text divided into small chunks for better relevant sentences
locating. In systhesis store, documents text divided into large chunks to provide sufficient LLM input content.

        python rag/advanced_rag_child_parent_retrieval.py

### 3.2 




