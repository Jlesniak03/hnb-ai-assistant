from dotenv import load_dotenv
import os

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Load your API key
load_dotenv()

# Load vector store
db = FAISS.load_local("vectorstore/products_faiss", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# Set up retriever
retriever = db.as_retriever()

# Set up QA chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=retriever
)

# Ask your question
query = input("Ask a question about a product: ")
response = qa.run(query)

print("\n🧠 Answer:\n", response)
