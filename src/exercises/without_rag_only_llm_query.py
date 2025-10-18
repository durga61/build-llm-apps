import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

def main():
    """
    Executes a query using a language model (LLM) with OpenAI embeddings and prints the response.
    Initializes OpenAI embeddings and a chat-based LLM, constructs a prompt from a template using the given query,
    invokes the LLM chain, and prints the generated response content.
    Note:
    - The query used is "what is Pinecone in machine learning?".
    - Assumes necessary imports and API keys are configured for OpenAIEmbeddings and ChatOpenAI.
    """
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    query = "what is Pinecone in machine learning?"
    chain = PromptTemplate.from_template(template=query) | llm
    response =  chain.invoke(input={})
    print(response.content)

if __name__ == "__main__":
    main()
    print("Loading, splitting, embedding, and storing document...")