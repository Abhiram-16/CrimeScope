from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import sys

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agent_ollama import run_agent

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# HTML template as a string (we'll serve it directly)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crime Forecaster AI - Intelligent Crime Analysis</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 20px;
            text-align: center;
            color: white;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .header h1 {
            font-size: 28px;
            margin-bottom: 5px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 14px;
        }
        
        .main-container {
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .chat-container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            width: 100%;
            max-width: 800px;
            height: 70vh;
            min-height: 500px;
            display: flex;
            flex-direction: column;
        }
        
        .stats-bar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 20px 20px 0 0;
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .stat-item {
            text-align: center;
        }
        
        .stat-value {
            font-size: 18px;
            font-weight: bold;
        }
        
        .stat-label {
            font-size: 12px;
            opacity: 0.9;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }
        
        .message {
            margin-bottom: 15px;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .user-message {
            text-align: right;
        }
        
        .user-message .bubble {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            display: inline-block;
            padding: 12px 18px;
            border-radius: 18px 18px 4px 18px;
            max-width: 70%;
            word-wrap: break-word;
            text-align: left;
        }
        
        .bot-message .bubble {
            background: white;
            color: #333;
            display: inline-block;
            padding: 12px 18px;
            border-radius: 18px 18px 18px 4px;
            max-width: 70%;
            word-wrap: break-word;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            line-height: 1.5;
        }
        
        .bot-message .bubble ul {
            margin: 10px 0 10px 20px;
        }
        
        .typing-indicator {
            display: none;
            padding: 12px 18px;
            background: white;
            border-radius: 18px;
            display: inline-block;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .typing-indicator.show {
            display: inline-block;
        }
        
        .typing-indicator span {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #999;
            margin: 0 2px;
            animation: bounce 1.4s infinite ease-in-out;
        }
        
        .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
        .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
            40% { transform: scale(1); opacity: 1; }
        }
        
        .chat-input {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
            border-radius: 0 0 20px 20px;
        }
        
        .input-form {
            display: flex;
            gap: 10px;
        }
        
        .input-form input {
            flex: 1;
            padding: 12px 18px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        
        .input-form input:focus {
            border-color: #667eea;
        }
        
        .input-form button {
            padding: 12px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s;
            font-weight: 500;
        }
        
        .input-form button:hover:not(:disabled) {
            transform: scale(1.05);
        }
        
        .input-form button:active:not(:disabled) {
            transform: scale(0.95);
        }
        
        .input-form button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .quick-actions {
            margin-top: 10px;
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        
        .quick-action {
            background: #f0f0f0;
            padding: 6px 12px;
            border-radius: 15px;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid transparent;
        }
        
        .quick-action:hover {
            background: #e0e0e0;
            border-color: #667eea;
        }
        
        .info-panel {
            background: #f8f9fa;
            padding: 15px;
            margin: 20px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        
        .info-panel h3 {
            color: #333;
            margin-bottom: 10px;
            font-size: 16px;
        }
        
        .info-panel ul {
            list-style: none;
            color: #666;
            font-size: 14px;
            line-height: 1.8;
        }
        
        .info-panel ul li:before {
            content: "→ ";
            color: #667eea;
            font-weight: bold;
        }
        
        @media (max-width: 768px) {
            .chat-container {
                height: 80vh;
                border-radius: 0;
            }
            
            .stats-bar {
                border-radius: 0;
            }
            
            .header h1 {
                font-size: 24px;
            }
            
            .quick-actions {
                justify-content: center;
            }
        }
        
        .error-message {
            background: #fee;
            color: #c33;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 10px;
            font-size: 14px;
        }
        
        .success-message {
            background: #efe;
            color: #3c3;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 10px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚔 Crime Forecaster AI</h1>
        <p>Powered by Machine Learning & Historical Crime Data Analysis</p>
    </div>
    
    <div class="main-container">
        <div class="chat-container">
            <div class="stats-bar">
                <div class="stat-item">
                    <div class="stat-value">5</div>
                    <div class="stat-label">Districts</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">10K+</div>
                    <div class="stat-label">Records</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">2006-2024</div>
                    <div class="stat-label">Data Range</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">ML</div>
                    <div class="stat-label">Powered</div>
                </div>
            </div>
            
            <div class="chat-messages" id="chat-messages">
                <div class="info-panel">
                    <h3>Welcome to Crime Forecaster AI!</h3>
                    <ul>
                        <li>Ask about crime data in Districts A, B, C, D, or E</li>
                        <li>Get predictions for future crime incidents</li>
                        <li>Search by crime type (assault, robbery, murder, etc.)</li>
                        <li>Analyze crime patterns and trends</li>
                    </ul>
                </div>
                <div class="message bot-message">
                    <div class="bubble">
                        Hello! I'm your AI crime analysis assistant. I can help you understand crime patterns and predict future incidents based on historical data. What would you like to know?
                    </div>
                </div>
            </div>
            
            <div class="chat-input">
                <form class="input-form" id="chat-form">
                    <input type="text" id="user-input" placeholder="Ask about crime data or predictions..." required autocomplete="off">
                    <button type="submit" id="send-btn">Send</button>
                </form>
                <div class="quick-actions">
                    <div class="quick-action" onclick="setExample('What crimes happened in District C?')">📍 District C Analysis</div>
                    <div class="quick-action" onclick="setExample('Predict crime for next 7 days')">📈 7-Day Forecast</div>
                    <div class="quick-action" onclick="setExample('Show me all murders')">🔍 Murder Cases</div>
                    <div class="quick-action" onclick="setExample('Which district has most crimes?')">📊 Crime Statistics</div>
                    <div class="quick-action" onclick="setExample('Show recent assaults')">⚠️ Recent Assaults</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const messagesDiv = document.getElementById('chat-messages');
        const chatForm = document.getElementById('chat-form');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        
        function formatMessage(text) {
            // Convert line breaks to <br> for better formatting
            return text.replace(/\\n/g, '<br>');
        }
        
        function addMessage(text, isUser) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
            
            const bubble = document.createElement('div');
            bubble.className = 'bubble';
            bubble.innerHTML = formatMessage(text);
            
            messageDiv.appendChild(bubble);
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function showTyping() {
            const typingDiv = document.createElement('div');
            typingDiv.className = 'message bot-message';
            typingDiv.id = 'typing';
            typingDiv.innerHTML = `
                <div class="typing-indicator show">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            `;
            messagesDiv.appendChild(typingDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function hideTyping() {
            const typing = document.getElementById('typing');
            if (typing) typing.remove();
        }
        
        function setExample(text) {
            userInput.value = text;
            userInput.focus();
        }
        
        function setLoading(loading) {
            sendBtn.disabled = loading;
            userInput.disabled = loading;
            sendBtn.textContent = loading ? 'Thinking...' : 'Send';
        }
        
        chatForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const question = userInput.value.trim();
            if (!question) return;
            
            // Add user message
            addMessage(question, true);
            userInput.value = '';
            
            // Show typing indicator and disable input
            showTyping();
            setLoading(true);
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query: question })
                });
                
                hideTyping();
                setLoading(false);
                
                if (response.ok) {
                    const data = await response.json();
                    addMessage(data.response, false);
                } else {
                    const error = await response.json();
                    addMessage('Sorry, I encountered an error: ' + (error.error || 'Unknown error'), false);
                }
            } catch (error) {
                hideTyping();
                setLoading(false);
                console.error('Error:', error);
                addMessage('Sorry, I could not process your request. Please check if the server is running.', false);
            }
        });
        
        // Auto-focus on input
        userInput.focus();
        
        // Add keyboard shortcut (Ctrl+Enter to send)
        userInput.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                chatForm.dispatchEvent(new Event('submit'));
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    """Serve the chat interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "Crime Forecaster AI"})

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