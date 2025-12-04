import os
import faiss
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, StoppingCriteria, StoppingCriteriaList
from sentence_transformers import SentenceTransformer

from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import LLMChain

from src.forecasting import predict_future

print("Waking up the main brain (LLM)...")
model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# ✅ Add END_OF_ANSWER token only if it's missing
if "END_OF_ANSWER" not in tokenizer.get_vocab():
    special_tokens_dict = {'additional_special_tokens': ['END_OF_ANSWER']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

# Custom stopping criteria
class StopOnSequence(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids
        self.buffer = []

    def __call__(self, input_ids, scores, **kwargs):
        self.buffer.append(int(input_ids[0, -1]))
        if len(self.buffer) > len(self.stop_ids):
            self.buffer.pop(0)
        return self.buffer == self.stop_ids

stop_ids = tokenizer.encode("END_OF_ANSWER", add_special_tokens=False)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    pad_token_id=tokenizer.eos_token_id,
    stopping_criteria=StoppingCriteriaList([StopOnSequence(stop_ids)])
)

llm = HuggingFacePipeline(pipeline=pipe)

print("Main brain is awake!")

print("Preparing the Diary (Knowledge Base)...")
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
    print(f"[TOOL LOG] Crime_Retriever called with query: {query}")
    query_emb = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_emb).astype("float32"), k)
    results = []
    for idx in indices[0]:
        if idx != -1 and idx < len(documents):
            results.append(documents[idx])
    if results:
        print(f"[TOOL LOG] Crime_Retriever returning {len(results)} results")
    else:
        print("[TOOL LOG] Crime_Retriever found no matching records")
    return "\n".join(results) if results else "No matching past records found."

print("Preparing the Crystal Ball (Forecasting Model)...")
def forecast_tool_wrapper(days: int):
    print(f"[TOOL LOG] Crime_Forecaster called for {days} days")
    result = predict_future(days)
    print(f"[TOOL LOG] Crime_Forecaster result: {result}")
    return f"Predicted {result[1]:.2f} incidents after {days} days (last data date: {result[0]})"

tools = [
    Tool(
        name="Crime_Retriever",
        func=retrieve_from_faiss,
        description="For any question about past crime events. Provide detailed crime incident data."
    ),
    Tool(
        name="Crime_Forecaster",
        func=forecast_tool_wrapper,
        description="For predicting future crime incidents over a number of days."
    )
]

prefix = """
You are a crime data assistant with access to two tools:

- Crime_Retriever(query: str) — retrieves detailed past crime events.
- Crime_Forecaster(days: int) — predicts future crime incidents.

RULES (follow exactly):
1. Answer ONLY the single user question given.
2. Use ONLY the exact information returned by the tools — copy verbatim from the Observation.
3. DO NOT invent names, dates, or details not explicitly in the Observation.
4. NEVER answer more than one question at a time.
5. The final line of your response MUST be exactly:
END_OF_ANSWER
6. If you omit END_OF_ANSWER, your answer will be considered invalid.

OUTPUT FORMAT (copy exactly):
Question: <repeat the user question>
Thought: <your reasoning>
Action: <Crime_Retriever or Crime_Forecaster>
Action Input: <parameters>
Observation: <full result from the tool>
Thought: <final reasoning>
Final Answer: <detailed and complete answer based on Observation>

END_OF_ANSWER

NOTES:
- Do NOT add explanations or anything after END_OF_ANSWER.
- Do NOT change capitalization or wording of END_OF_ANSWER.
"""

suffix = "\nQuestion: {input}\n{agent_scratchpad}"

prompt = PromptTemplate(
    template=prefix + suffix,
    input_variables=["input", "agent_scratchpad"]
)

llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=1
)
#agent log recovery and chat history additions in progress
print("✅ Robot is ready to chat with its final brain!")

def run_agent(user_input: str):
    print(f"[AGENT LOG] Received query: {user_input}")
    try:
        response_dict = agent_executor.invoke(
            {"input": user_input}
        )

        if isinstance(response_dict, dict):
            response = response_dict.get("output", "")
        else:
            response = str(response_dict)

        if "END_OF_ANSWER" not in response:
            print("[AGENT WARNING] Missing END_OF_ANSWER — rejecting response.")
            return "Error: Incomplete answer detected. Please try rephrasing your question."

        # Trim output at END_OF_ANSWER
        response = response.split("END_OF_ANSWER")[0].strip()
          
        placeholders = ["[Date]", "[Location]", "[Details]", "[Type of Crime]"]
        if any(ph in response for ph in placeholders):
            print("[AGENT WARNING] Placeholder detected in response.")
            return "Error: Incomplete answer detected. Please try rephrasing your question."

        if response.count("Question:") > 1:
            print("[AGENT WARNING] Multiple questions detected in response.")
            return "Error: Multiple questions detected. Please ask only one question at a time."

        if "Final Answer:" not in response:
            print("[AGENT WARNING] No final answer detected in response.")
            return "Error: No final answer detected. Please try rephrasing your question."

    except ValueError as ve:
        print(f"[AGENT ERROR] {ve}")
        return "Error: The assistant tried to answer multiple questions. Please ask only one at a time."
    except Exception as e:
        print(f"[AGENT ERROR] Unexpected error: {e}")
        return f"Error: {str(e)}"

    print(f"[AGENT LOG] Final response: {response}")
    return response

