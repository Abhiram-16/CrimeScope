# src/data_processor.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# --- Configuration ---
CSV_PATH = "data/crime_data.csv"
VECTOR_STORE_DIR = "vector_store"
VECTOR_STORE_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index")

# The corrected column names we discovered
TEXT_COLUMNS = [
    'Incident Type Primary',
    'Incident Datetime',
    'Day of Week',
    'Police District'
]

def create_knowledge_base():
    """
    Loads data from the CSV, creates text embeddings, and saves them to a FAISS index.
    This function encapsulates the logic from the '1_Data_Processing.ipynb' notebook.
    """
    print("--- Starting Knowledge Base creation ---")
    
    # 1. Load and Prepare Data
    print("Step 1: Loading and preparing data...")
    if not os.path.exists(CSV_PATH):
        print(f"Error: The file {CSV_PATH} was not found.")
        return

    df = pd.read_csv(CSV_PATH)
    df.dropna(subset=TEXT_COLUMNS, inplace=True)
    for col in TEXT_COLUMNS:
        df[col] = df[col].astype(str)

    # 2. Create Text Documents
    documents = []
    for _, row in df.iterrows():
        doc = f"Incident Type: {row['Incident Type Primary']} on {row['Day of Week']}, {row['Incident Datetime']}. Police District: {row['Police District']}."
        documents.append(doc)
    print(f"Created {len(documents)} documents.")

    # 3. Create Embeddings
    print("Step 2: Creating text embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents, show_progress_bar=True)

    # 4. Build and Save FAISS Index
    print("Step 3: Building and saving the FAISS index...")
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    faiss.write_index(index, f"{VECTOR_STORE_PATH}.index")

    with open(f"{VECTOR_STORE_PATH}.docs", "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc + "\n")

    print(f"\n✅ Success! Knowledge base created and saved in the '{VECTOR_STORE_DIR}' directory.")


if __name__ == '__main__':
    # This block allows us to run this script directly from the command line
    create_knowledge_base()