from dotenv import load_dotenv
import os

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Load the saved ailments vector store
db = FAISS.load_local("vectorstore/ailments_faiss", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# Set up retriever
retriever = VectorStoreRetriever(vectorstore=db)

# Set up retrieval QA chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4", temperature=0),
    retriever=retriever
)

# Ask a question about an ailment
query = input("Ask a question about an ailment: ")
response = qa.run(query)
print("\n🤖 Answer:\n", response)
