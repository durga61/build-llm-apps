import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

# --- 1. Load all PDFs from directory ---
# Loading documents
# Let’s load a PDF into a sequence of Document objects, PyPDFLoader loads one Document object per PDF page. For each, we can easily access:
# The string content of the page;
# Metadata containing the file name and page number.
# print(f"{docs[0].page_content[:200]}\n")
# print(docs[0].metadata) 

def load_pdfs_from_dir(pdf_dir: str):
    all_docs = []
    for file in glob.glob(os.path.join(pdf_dir, "*.pdf")):
        print(f"Loading {file}...")
        loader = PyPDFLoader(file, password=os.environ['CREDIT_CARD_PDF_PASSWORD'])
        docs = loader.load()
        # add filename as metadata (useful for month filtering later)
        for d in docs:
            d.metadata["source_file"] = os.path.basename(file)
        all_docs.extend(docs)
        print(f"Loaded {len(docs)} pages from {file}")
        print(f"Total documents so far: {len(all_docs)}")
    return all_docs

def main():
    """
    Loads a PDF document, splits its content into text chunks, generates embeddings for each chunk using OpenAI's embedding model, and stores the embeddings in a Pinecone vector index for semantic search.
    """
    #Splitting
    #For both information retrieval and downstream question-answering purposes, a page may be too coarse a representation. 
    # Our goal in the end will be to retrieve Document objects that answer an input query, and further splitting our PDF will help ensure that the meanings of relevant portions 
    # of the document are not “washed out” by surrounding text.
    #We can use text splitters for this purpose. Here we will use a simple text splitter that partitions based on characters. We will split our documents into chunks of 1000 characters
    #with 200 characters of overlap between chunks. The overlap helps mitigate the possibility of separating a statement from important context related to it.
    #  We use the RecursiveCharacterTextSplitter, which will recursively split the document using common separators like new lines until each chunk is the appropriate size.
    #  This is the recommended text splitter for generic text use cases.
    # file_path = "../../creditcard/axis/Credit Card Statement.pdf"
    # loader = PyPDFLoader(file_path, password=os.environ['CREDIT_CARD_PDF_PASSWORD'])
    # document = loader.load()
    document = load_pdfs_from_dir("../../creditcard/axis/")
    print(f"Number of documents: {len(document)}")


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    all_splits = text_splitter.split_documents(document)
    print(f"Number of chunks: {len(all_splits)}")
    #Embeddings
    #Vector search is a common way to store and search over unstructured data (such as unstructured text). 
    # The idea is to store numeric vectors that are associated with the text. Given a query, we can embed it as a vector of the same dimension 
    # and use vector similarity metrics (such as cosine similarity) to identify related text.
    #LangChain supports embeddings from dozens of providers. These models specify how text should be converted into a numeric vector. Let’s select a model:
    OpenAI_Embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    # vector1 = OpenAI_Embeddings.embed_query(all_splits[0].page_content)
    # print(f"Generated vectors of length {len(vector1)}\n")
    #print(vector1)
    #Vector stores
    #pc = Pinecone()
    #Vector stores
    #Armed with a model for generating text embeddings, we can next store them in a special data structure that supports efficient similarity search
    #LangChain @[VectorStore] objects contain methods for adding text and Document objects to the store,
    # and querying them using various similarity metrics. They are often initialized with embedding models, which determine how text data is translated to numeric vectors.
    #LangChain includes a suite of integrations with different vector store technologies.
    # Several of these are hosted services, which means you can get started with them quickly without installing any software.
    # Here we will use Pinecone, a popular vector database that is available as a hosted service.
    #To use Pinecone, you will need to create a free account and get an API key. You will also need to create an index, which is where your vectors will be stored.
    # You can do this from the Pinecone console. Make sure to select a dimension that matches the dimension of the embedding model you selected earlier (1536 for text-embedding-3-large).
    #Once you have your API key and index name, you can set them as environment variables   PINECONE_API_KEY and PC_INDEX respectively.
    # You can then initialize a Pinecone vector store as follows:   
    PineconeVectorStore.from_documents(all_splits, OpenAI_Embeddings, index_name=os.environ['PC_INDEX'])


if __name__ == "__main__":
    main()
