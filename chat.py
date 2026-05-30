from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

load_dotenv()

embeddings = OpenAIEmbeddings(
   model="text-embedding-3-small"
)

print("Loading FAISS index...")

vector_store = FAISS.load_local(
   "faiss_index",
   embeddings,
   allow_dangerous_deserialization=True
)

print("Vector database loaded")

user_query = input("\nAsk a question: ")

results = vector_store.similarity_search(
   user_query,
   k=3   
)

context = "\n\n".join(
    [doc.page_content for doc in results]
)

llm = ChatOpenAI(
   model="gpt-4o-mini",
    temperature=0
)

prompt = f"""
Answer the question using ONLY the provided context.

Context:
{context}

Question:
{user_query}

If the answer is not present in the context, say:
"I could not find the answer in the provided document."
"""

response = llm.invoke(prompt)

print("\nAnswer:")
print(response.content)