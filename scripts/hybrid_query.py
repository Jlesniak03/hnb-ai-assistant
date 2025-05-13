from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Load both vectorstores
product_db = FAISS.load_local("vectorstore/products_faiss", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
ailment_db = FAISS.load_local("vectorstore/ailments_faiss", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

product_retriever = VectorStoreRetriever(vectorstore=product_db)
ailment_retriever = VectorStoreRetriever(vectorstore=ailment_db)

# Set up GPT-4 model
gpt4 = ChatOpenAI(model="gpt-4", temperature=0)

# QA chains
product_qa = RetrievalQA.from_chain_type(llm=gpt4, retriever=product_retriever)
ailment_qa = RetrievalQA.from_chain_type(llm=gpt4, retriever=ailment_retriever)

# Detect topic type based on keyword (basic method for now)
def is_ailment_question(query):
    keywords = ["help with", "feel", "pain", "tired", "symptoms", "treatment", "relief", "issue", "condition"]
    return any(word in query.lower() for word in keywords)

# Ask the user
query = input("Ask your question: ")

# Choose vectorstore
if is_ailment_question(query):
    print("🔍 Using AILMENT knowledge base...")
    response = ailment_qa.run(query)
    source = "internal (ailments)"
else:
    print("🔍 Using PRODUCT knowledge base...")
    response = product_qa.run(query)
    source = "internal (products)"

# Trigger fallback if the internal answer is vague or unhelpful
failure_keywords = [
    "i don't know", "not sure", "not included",
    "no information", "i'm sorry", "does not include details"
]
if not response or any(kw in response.lower() for kw in failure_keywords) or len(response.strip()) < 60:
    print("⚠️ Falling back to GPT-4 browsing...")
    fallback_prompt = f"Answer this question using your own external knowledge or live browsing: '{query}'"
    response = gpt4.invoke(fallback_prompt).content
    source = "fallback (GPT-4 w/ browsing)"

# Show result
print(f"\n🤖 Answer ({source}):\n{response}")
