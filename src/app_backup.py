from flask import Flask, request, jsonify
from src.agent_ollama import run_agent

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "LLM Crime Forecaster API",
        "endpoints": {
            "/chat": "POST - Send a query to the chatbot",
            "/health": "GET - Check if service is running"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get("query")

    if not query:
        return jsonify({"error": "You need to send a question!"}), 400

    print(f"Received a question: {query}")

    try:
        response_text = run_agent(query)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    print(f"Robot's answer: {response_text}")

    return jsonify({"response": response_text})

if __name__ == '__main__':
    print("Starting Flask server on http://127.0.0.1:5001")
    app.run(host='127.0.0.1', port=5001, debug=True)  # Changed to port 5001