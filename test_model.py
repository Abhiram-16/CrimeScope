#!/usr/bin/env python3
"""
Test with TinyLlama - a model that doesn't have cache issues
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

print("Testing with TinyLlama (1.1B params)...")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    temperature=0.1,
    pad_token_id=tokenizer.eos_token_id
)

# Test 1: Agent format test
test_prompt_1 = """You have access to a search tool.

Format:
Action: search
Action Input: query text
Observation: [search results appear here]
Final Answer: [your answer]

Question: What happened downtown?
Action: search
Action Input:"""

print("\n" + "="*60)
print("TEST: Does TinyLlama stop or hallucinate?")
print("="*60)
print("Prompt ends with: '...Action Input:'")
print("-"*40)
print("Model continues with:")
print("-"*40)

result = pipe(test_prompt_1)[0]['generated_text']
continuation = result[len(test_prompt_1):]
print(continuation)

print("\n" + "="*60)
print("ANALYSIS:")
print("="*60)

if "Observation:" in continuation or "[search results" in continuation:
    print("❌ Model hallucinates the Observation!")
    print("   Even TinyLlama struggles with agent tasks.")
    print("\nRECOMMENDATION: Use Mistral-7B or Llama-3 with Ollama")
else:
    print("✅ Model stopped appropriately!")
    print("   TinyLlama might work for your agent.")

# Also test Phi-3 comparison
print("\n" + "="*60)
print("For comparison with Phi-3:")
print("- TinyLlama: 1.1B parameters")
print("- Phi-3-mini: 3.8B parameters")
print("- If TinyLlama works better, it's not just about size!")
print("="*60)