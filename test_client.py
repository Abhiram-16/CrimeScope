# test_client.py
import requests
import json

# The address of our chatbot
url = "http://127.0.0.1:5000/chat"

# The question we want to ask
query = "what happened in the Downtown police district?"

print("Sending question to the chatbot...")

headers = {"Content-Type": "application/json"}
data = {"query": query}

try:
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        print("\n--- Chatbot's Answer ---")
        print(response.json()['response'])
    else:
        print(f"\n--- Error ---")
        print(f"Status Code: {response.status_code}")
        print(response.text)

except requests.exceptions.ConnectionError as e:
    print("\n--- Connection Error ---")
    print(f"Could not connect to the server at {url}.")
    print("Please make sure your Flask application is running in the other terminal.")