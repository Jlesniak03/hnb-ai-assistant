import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

# ✅ Load environment from /app/.env (explicit path)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', 'app', '.env'))

# Paths
product_text_path = os.path.join("..", "app", "Data", "product_documents.txt")
ailment_text_path = os.path.join("..", "app", "Data", "ailment_documents.txt")

product_output_path = os.path.join("..", "app", "vectorstore", "products_faiss")
ailment_output_path = os.path.join("..", "app", "vectorstore", "ailments_faiss")

# Embeddings
embeddings = OpenAIEmbeddings()

# Text splitter
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Load, split, embed, save for products and ailments
def build_vectorstore(input_path, output_path):
    print(f"Building vectorstore from {input_path} -> {output_path}...")
    loader = TextLoader(input_path, encoding="utf-8")
    docs = loader.load()
    splits = splitter.split_documents(docs)
    db = FAISS.from_documents(splits, embeddings)
    db.save_local(output_path)
    print("✅ Done.")

if __name__ == "__main__":
    build_vectorstore(product_text_path, product_output_path)
    build_vectorstore(ailment_text_path, ailment_output_path)