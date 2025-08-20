const messagesDiv = document.getElementById('messages');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

// Auto-resize textarea
userInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

function addMessage(text, isUser) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
    
    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = text.replace(/\n/g, '<br>');
    
    messageDiv.appendChild(content);
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function showTyping() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot';
    typingDiv.id = 'typing';
    typingDiv.innerHTML = `
        <div class="typing">
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
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
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
    userInput.focus();
}

function setLoading(loading) {
    sendBtn.disabled = loading;
    userInput.disabled = loading;
    sendBtn.innerHTML = loading ? '⏳' : '➤';
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const question = userInput.value.trim();
    if (!question) return;
    
    addMessage(question, true);
    userInput.value = '';
    userInput.style.height = 'auto';
    
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

// Handle Enter key
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
});

userInput.focus();