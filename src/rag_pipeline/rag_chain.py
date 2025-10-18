# Rag Chain
# A retrieval-augmented generation (RAG) chain that combines a language model with a vector store to provide context-aware responses based on relevant documents.
# Embedding user Query
# Semantic Search in Vector DB
# Context + Query -> LLM -> Response (Prompt Augmentation)


import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

def main():
    """
    Executes a query using a language model (LLM) with OpenAI embeddings and prints the response.
    Initializes OpenAI embeddings and a chat-based LLM, constructs a prompt from a template using the given query,
    invokes the LLM chain, and prints the generated response content.
    Note:
    - Assumes necessary imports and API keys are configured for OpenAIEmbeddings and ChatOpenAI.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    llm = ChatOpenAI()

    vector_store = PineconeVectorStore(embedding=embeddings, index_name=os.environ['PC_INDEX'])
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever=vector_store.as_retriever(), combine_docs_chain=combine_docs_chain)
    queries = [
    "Summarize monthly spending across all months.",
    "List top 10 merchants by total spend across the months.",
    "Find recurring subscriptions that appear every month.",
    "Identify anomalies: unusually large purchases compared to monthly averages."
    ]

    for q in queries:
        response = retrieval_chain.invoke({"input": q})
        print("Q:", q)
        print("A:", response["answer"])
        print("="*60)
    #result = retrieval_chain.invoke(input={"input": queries})
    #print(result["answer"])
if __name__ == "__main__":
    main()
