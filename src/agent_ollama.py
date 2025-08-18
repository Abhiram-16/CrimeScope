import os
import sys
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Fix import path for forecasting
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from forecasting import predict_future

print("Initializing Ollama LLM...")
# THIS IS THE MAIN CHANGE - Using Ollama instead of HuggingFace
llm = OllamaLLM(
    model="mistral:7b-instruct-q4_K_M",  # or "llama3.2:3b-instruct-q8_0" for faster
    temperature=0.3,
    num_predict=512,  # max tokens to generate
    num_ctx=4096,     # context window
    verbose=True
)

print("Testing LLM connection...")
try:
    test_response = llm.invoke("Say 'ready'")
    print(f"LLM Test Response: {test_response}")
    print("✅ Ollama LLM is ready!")
except Exception as e:
    print(f"❌ Error connecting to Ollama: {e}")
    print("Make sure 'ollama serve' is running in another terminal!")
    exit(1)

# ========== KNOWLEDGE BASE SETUP (Same as before) ==========
print("Preparing the Knowledge Base...")
base_dir = os.path.dirname(os.path.abspath(__file__))
faiss_index_path = os.path.join(base_dir, "..", "vector_store", "faiss_index.index")
docs_store_path = os.path.join(base_dir, "..", "vector_store", "faiss_index.docs")

if not os.path.exists(faiss_index_path):
    raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")
if not os.path.exists(docs_store_path):
    raise FileNotFoundError(f"Document store not found at {docs_store_path}")

index = faiss.read_index(faiss_index_path)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

with open(docs_store_path, "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f if line.strip()]

def retrieve_from_faiss(query: str, k: int = 5):
    """Search past crime events in the knowledge base"""
    print(f"[TOOL LOG] Crime_Retriever called with query: {query}")
    
    # Handle common district queries
    query_lower = query.lower()
    
    # Map common names to actual district names
    # Only map if it's a standalone word, not part of "District D"
    district_mapping = {
        "downtown": "District C",  # Assuming District C is downtown
        "central": "District C",
        "north": "District A",
        "south": "District E",
        "east": "District B",
        "west": "District D",
    }
    
    # Check if query already contains "District" (like "District D")
    if "district" not in query_lower:
        # Only then check for single letter districts
        single_letter_mapping = {
            "a": "District A",
            "b": "District B", 
            "c": "District C",
            "d": "District D",
            "e": "District E"
        }
        
        # Check if query is asking for a specific district
        for common_name, actual_district in district_mapping.items():
            if common_name in query_lower:
                query = actual_district
                print(f"[TOOL LOG] Mapped '{common_name}' to '{actual_district}'")
                break
        
        # Check for single letter districts
        if query_lower.strip() in single_letter_mapping:
            query = single_letter_mapping[query_lower.strip()]
            print(f"[TOOL LOG] Mapped '{query_lower.strip()}' to '{query}'")
    
    query_emb = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_emb).astype("float32"), k)
    results = []
    for idx in indices[0]:
        if idx != -1 and idx < len(documents):
            results.append(documents[idx])
    
    # If searching for a specific district, filter results
    if "District" in query:
        filtered_results = [r for r in results if query in r]
        if filtered_results:
            results = filtered_results
            print(f"[TOOL LOG] Filtered to {len(results)} results for {query}")
    
    if results:
        print(f"[TOOL LOG] Crime_Retriever returning {len(results)} results")
        return "\n".join(results)
    else:
        print("[TOOL LOG] Crime_Retriever found no matching records")
        return "No matching past records found."

# ========== FORECASTING SETUP (Same as before) ==========
print("Preparing the Forecasting Model...")
def forecast_tool_wrapper(days: str):
    """Predict future crime incidents for specified number of days"""
    try:
        # Handle both string and int input
        if isinstance(days, str):
            # Extract number from string like "7 days" or just "7"
            days = int(''.join(filter(str.isdigit, days.split()[0])))
        else:
            days = int(days)
    except (ValueError, IndexError):
        return "Error: Please provide a valid number of days (e.g., '7' or '7 days')"
    
    print(f"[TOOL LOG] Crime_Forecaster called for {days} days")
    try:
        result = predict_future(days)
        print(f"[TOOL LOG] Crime_Forecaster result: {result}")
        return f"Predicted {result[1]:.2f} incidents after {days} days (last data date: {result[0]})"
    except Exception as e:
        print(f"[TOOL LOG] Forecasting error: {e}")
        return f"Error in forecasting: {str(e)}"

# ========== TOOLS DEFINITION ==========
tools = [
    Tool(
        name="Crime_Retriever",
        func=retrieve_from_faiss,
        description="Search for past crime events. Use this when asking about what happened, past incidents, or historical crime data. Input should be a search query."
    ),
    Tool(
        name="Crime_Forecaster",
        func=forecast_tool_wrapper,
        description="Predict future crime incidents. Use this when asking about predictions or future crime trends. Input should be number of days."
    )
]

# ========== OPTIMIZED PROMPT FOR MISTRAL ==========
# Mistral works better with clear, concise prompts
prefix = """You are a helpful crime data assistant with access to tools.

Available tools:
{tools}

To answer questions, use this EXACT format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: [EXACTLY one of: Crime_Retriever OR Crime_Forecaster]
Action Input: the input to the action
Observation: the result of the action
Thought: I now have the information I need
Final Answer: the final answer to the original input question

CRITICAL RULES:
1. In the Action line, write ONLY the tool name, nothing else!
2. After receiving an Observation, provide your Final Answer immediately
3. Do NOT repeat the same action twice
4. When you have crime data from the Observation, summarize it in your Final Answer
5. For vague questions about crime rates, use Crime_Forecaster to predict daily averages

Important:
- For past events, use Crime_Retriever
- For predictions or rates, use Crime_Forecaster
- Be specific in your Action Input

Begin!

"""

suffix = """Question: {input}
{agent_scratchpad}"""

prompt = PromptTemplate(
    template=prefix + suffix,
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
)

# ========== AGENT SETUP ==========
print("Setting up the agent...")
llm_chain = LLMChain(llm=llm, prompt=prompt)

agent = ZeroShotAgent(
    llm_chain=llm_chain,
    tools=tools,
    verbose=True
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=2,  # Reduced from 3 to prevent loops
    handle_parsing_errors=True,
    early_stopping_method="force",  # Force stop after max iterations
    return_intermediate_steps=False
)

print("✅ Agent is ready to chat!")

# ========== RUN AGENT FUNCTION ==========
def run_agent(user_input: str):
    """Process user input and return agent response"""
    print(f"[AGENT LOG] Received query: {user_input}")
    
    try:
        # Format tools for the prompt
        tools_desc = "\n".join([f"- {t.name}: {t.description}" for t in tools])
        tool_names = ", ".join([t.name for t in tools])
        
        # Run the agent
        response_dict = agent_executor.invoke({
            "input": user_input,
            "tools": tools_desc,
            "tool_names": tool_names
        })
        
        # Extract the response
        if isinstance(response_dict, dict):
            response = response_dict.get("output", "")
        else:
            response = str(response_dict)
        
        # Clean up the response - extract just the final answer
        if "Final Answer:" in response:
            # Get everything after "Final Answer:"
            final_answer = response.split("Final Answer:")[-1].strip()
            # Remove any trailing format markers
            if "Question:" in final_answer:
                final_answer = final_answer.split("Question:")[0].strip()
            response = final_answer
        
        # Validate response
        if not response or len(response) < 10:
            return "I couldn't generate a complete answer. Please try rephrasing your question."
        
        print(f"[AGENT LOG] Final response length: {len(response)} chars")
        return response
        
    except Exception as e:
        import traceback
        print(f"[AGENT ERROR] Unexpected error: {e}")
        print(f"[AGENT ERROR] Traceback:\n{traceback.format_exc()}")
        return "I encountered an error processing your request. Please try again."

# ========== TEST THE AGENT ==========
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing the agent with sample queries...")
    print("="*60)
    
# Remove the test code when importing as a module
# Keep it only when running directly for testing