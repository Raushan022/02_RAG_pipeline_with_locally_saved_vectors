from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

load_dotenv()

loader = PyPDFLoader("pdfs/nodejs.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = FAISS.from_documents(
   chunks,
   embeddings
)

# saving vector store to local
# vector_store.save_local("faiss_index")

print("vector db created")

user_query = input("Ask any question: ")

results = vector_store.similarity_search(user_query, k=3)
# print(len(results))
# print(type(results))
context = "\n\n".join(
      [doc.page_content for doc in results]
   )

llm = ChatOpenAI(
   model="gpt-4o-mini",
   temperature=0
)

prompt = f""""
Answer the question using ONLY the provided context.

context:
{context}

Question:
{user_query}

If the answer is not in the context, say
"I could not find the answer in the provided document."
"""

response = llm.invoke(prompt)
print(response.content)



