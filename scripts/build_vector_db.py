from dotenv import load_dotenv
import os

load_dotenv()  # ✅ Loads environment variables from .env

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Load your text documents
with open("./Data/product_documents.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
texts = text_splitter.split_text(text)

# Create embeddings using a valid model name
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create vector store
db = FAISS.from_texts(texts, embeddings)

# Save the vector store to disk
db.save_local("vectorstore/products_faiss")
print("✅ Vector store built and saved.")
