from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import sys

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agent_ollama import run_agent

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def home():
    """Serve the chat interface"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "CrimeScope AI"})

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint for the AI assistant"""
    data = request.json
    query = data.get("query")

    if not query:
        return jsonify({"error": "Please provide a question"}), 400

    print(f"Received question: {query}")

    try:
        response_text = run_agent(query)
        print(f"Response: {response_text[:100]}...")  # Log first 100 chars
        return jsonify({"response": response_text})
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Crime Forecaster AI is starting...")
    print("="*60)
    print("\n📱 Access the chatbot at: http://localhost:5001")
    print("🌐 Share this link for others to use (on same network)")
    print("\n" + "="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=False)