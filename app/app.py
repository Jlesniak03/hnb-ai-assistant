from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from datetime import datetime, timedelta
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA

# Load environment variables
if os.environ.get("RENDER") != "true":
    load_dotenv()


app = Flask(__name__)

# Session memory
chat_sessions = {}

# Load vectorstores once
product_db = FAISS.load_local("vectorstore/products_faiss", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
ailment_db = FAISS.load_local("vectorstore/ailments_faiss", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# QA chains
product_qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-4", temperature=0), retriever=VectorStoreRetriever(vectorstore=product_db))
ailment_qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-4", temperature=0), retriever=VectorStoreRetriever(vectorstore=ailment_db))
gpt4 = ChatOpenAI(model="gpt-4", temperature=0)

# Ailment keyword detection
def is_ailment_question(query):
    keywords = ["help with", "feel", "pain", "tired", "symptoms", "treatment", "relief", "issue", "condition"]
    return any(word in query.lower() for word in keywords)

# Cleanup old sessions
def cleanup_sessions():
    now = datetime.utcnow()
    to_delete = [k for k, v in chat_sessions.items() if now - v["last_active"] > timedelta(hours=2)]
    for k in to_delete:
        del chat_sessions[k]

# Kill switch helper
def kill_switch_active():
    # Prefer environment variable set on platform (e.g., Render) rather than .env
    return os.environ.get("KILL_SWITCH", "off").lower() == "on"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/send", methods=["POST"])
def send():
    if kill_switch_active():
        return jsonify({"error": "Service is temporarily disabled by the administrator."}), 503

    cleanup_sessions()
    data = request.get_json()
    tab_id = data.get("tab_id")
    query = data.get("query", "").strip()

    if not query or not tab_id:
        return jsonify({"error": "Missing tab_id or query"}), 400

    if tab_id not in chat_sessions:
        chat_sessions[tab_id] = {"messages": [], "last_active": datetime.utcnow()}

    session = chat_sessions[tab_id]
    session["last_active"] = datetime.utcnow()
    session["messages"].append({"role": "user", "content": query})

    if is_ailment_question(query):
        response = ailment_qa.run(query)
        source = "internal (ailments)"
    else:
        response = product_qa.run(query)
        source = "internal (products)"

    failure_keywords = ["i don't know", "not sure", "not included", "no information", "i'm sorry"]
    if not response or any(kw in response.lower() for kw in failure_keywords) or len(response.strip()) < 60:
        fallback_prompt = (
            "You are a trusted health advisor for Holland & Barrett. Only answer questions in the context of human health, "
            "supplements, nutrition, or wellness. Do not provide historical, manufacturing, or industrial information unless "
            "directly related to human use. Prioritize clear, practical guidance that supports energy, immunity, digestion, "
            "sleep, stress, and general wellbeing.\n\n"
            "Always recommend 2 to 3 specific Holland & Barrett supplements that are relevant to the customer's concern. "
            "Include their product names and explain how they help. If appropriate, briefly explain dietary or lifestyle tips. "
            "Keep your answer friendly, helpful, accurate, and under 120 words unless the topic needs more depth.\n\n"
            f"Customer query: {query} "
            "Use a warm, supportive tone — like a helpful staff member at a health store."
        )

        response = gpt4.invoke(fallback_prompt).content
        source = "fallback (GPT-4 w/ browsing)"

    session["messages"].append({"role": "assistant", "content": f"{response}\n\n🔍 Source: {source}"})
    return jsonify({
        "response": response,
        "source": source,
        "messages": session["messages"]
    })

@app.route("/delete_tab", methods=["POST"])
def delete_tab():
    tab_id = request.get_json().get("tab_id")
    if tab_id and tab_id in chat_sessions:
        del chat_sessions[tab_id]
        return jsonify({"success": True})
    return jsonify({"error": "Tab not found"}), 404

@app.route("/admin/toggle_kill", methods=["POST"])
def toggle_kill_switch():
    new_value = request.get_json().get("status")
    if new_value not in ["on", "off"]:
        return jsonify({"error": "Invalid status. Use 'on' or 'off'."}), 400

    os.environ["KILL_SWITCH"] = new_value  # Only affects runtime, not Render
    return jsonify({"message": f"Kill switch set to '{new_value}' (only during runtime, not persistent)."})

@app.route("/debug_kill")
def debug_kill():
    return jsonify({
        "KILL_SWITCH": os.environ.get("KILL_SWITCH"),
        "kill_switch_active": kill_switch_active()
    })
