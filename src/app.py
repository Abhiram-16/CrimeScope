from flask import Flask, request, jsonify
# Import the run_agent function from the updated agent file
from src.agent import run_agent

app = Flask(__name__)

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
    app.run(host='0.0.0.0', port=5000)
