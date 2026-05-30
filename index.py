from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

print("Loading PDF...")

loader = PyPDFLoader("pdfs/nodejs.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

embeddings = OpenAIEmbeddings(
   model="text-embedding-3-small"
)

print("Creating embeddings and FAISS index...")

vector_store = FAISS.from_documents(
   chunks,
   embeddings
)

vector_store.save_local("faiss_index")

print("FAISS index saved successfully!")