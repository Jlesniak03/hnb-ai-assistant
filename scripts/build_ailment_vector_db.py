from dotenv import load_dotenv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# ✅ Load environment variables (including OpenAI API key)
load_dotenv()

# ✅ Load your processed ailment documents
with open("./Data/ailment_documents.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ✅ Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
texts = text_splitter.split_text(text)

# ✅ Generate embeddings using OpenAI
embeddings = OpenAIEmbeddings()

# ✅ Build FAISS vector store
db = FAISS.from_texts(texts, embeddings)

# ✅ Save to disk
db.save_local("vectorstore/ailments_faiss")
print("✅ Ailment vector store built and saved.")
