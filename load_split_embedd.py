
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()

def main():
    #Loading documents
    file_path = "./docs/nke-10k-2023.pdf"
    pc_index= "pdf-search-embedding"
    loader = PyPDFLoader(file_path)
    document = loader.load()
    print(f"Number of documents: {len(document)}")
    #PyPDFLoader loads one Document object per PDF page. For each, we can easily access:
    # print(f"{document[0].page_content[:200]}\n")
    # print(document[0].metadata)
    #Splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    all_splits = text_splitter.split_documents(document)
    print(f"Number of chunks: {len(all_splits)}")
    #Embeddings
    OpenAI_Embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    #Eg: embed one chunk and see the vector
    vector1 = OpenAI_Embeddings.embed_query(all_splits[0].page_content)
    print(f"Generated vectors of length {len(vector1)}\n")
    print(vector1)


if __name__ == "__main__":
    main()
