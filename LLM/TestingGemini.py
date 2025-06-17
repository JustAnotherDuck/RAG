import time
import csv
import json
from datetime import datetime, timedelta
from google import genai
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import asyncio
import aiohttp
from ratelimit import limits, sleep_and_retry

# --- Configuration ---
CHROMA_DB_PATH = "chroma_db_semantic"
CHROMA_COLLECTION_NAME = "semantic_rag_collection_bge_v1.5"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
TOP_K = 5
API_KEY = "AIzaSyADiMx-RWGuuFT5amWb1VsZAfVO3kJysQk"
QUERY_FILE = "queries.txt"
LOG_FILE = "rag_test_log.csv"

# Rate limiting configuration
RPM_LIMIT = 15  # Requests per minute
TPM_LIMIT = 1000000  # Tokens per minute (approximate)
RPD_LIMIT = 1500  # Requests per day
CALLS = 15
PERIOD = 60  # 1 minute in seconds

# Initialize clients
client = genai.Client(api_key=API_KEY)
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
ch_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = ch_client.get_collection(name=CHROMA_COLLECTION_NAME)

# Rate limiting counters
request_count = 0
token_count = 0
day_start = datetime.now()
request_day_count = 0

# --- Read Queries from File ---
def load_queries(file_path):
    try:
        with open(file_path, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
        if len(queries) != 20:
            raise ValueError(f"Expected 20 queries, found {len(queries)}")
        return queries
    except Exception as e:
        print(f"Error reading queries file: {e}")
        return []

# --- Rate Limiting Decorator ---
@sleep_and_retry
@limits(calls=RPM_LIMIT, period=PERIOD)
def rate_limited_generate_content(prompt):
    global request_count, token_count, request_day_count, day_start
    
    # Check daily limit
    current_time = datetime.now()
    if (current_time - day_start).days >= 1:
        request_day_count = 0
        day_start = current_time
    
    if request_day_count >= RPD_LIMIT:
        raise Exception("Daily request limit reached")
    
    # Estimate tokens (rough approximation)
    tokens = len(prompt.split())  # Simple token estimation
    if token_count + tokens > TPM_LIMIT:
        raise Exception("Token per minute limit reached")
    
    # Update counters
    request_count += 1
    token_count += tokens
    request_day_count += 1
    
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=prompt
    )
    return response.text

# --- Process Single Query ---
async def process_query(query, iteration, log_writer):
    log_entry = {
        'query': query,
        'iteration': iteration,
        'embedding_time': 0,
        'retrieval_time': 0,
        'llm_time': 0,
        'response': '',
        'timestamp': datetime.now().isoformat()
    }
    
    # Embed query
    start_time = time.time()
    query_embedding = embed_model.encode([query])[0].tolist()
    log_entry['embedding_time'] = time.time() - start_time
    
    # Query ChromaDB
    start_time = time.time()
    results = collection.query(query_embeddings=[query_embedding], n_results=TOP_K)
    retrieved_chunks = results['documents'][0]
    log_entry['retrieval_time'] = time.time() - start_time
    
    # Construct prompt
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""You are a helpful assistant. Use the following context to answer the user's question.

Context:
{context}

Question:
{query}

Answer:"""
    
    # Query LLM
    start_time = time.time()
    try:
        response = rate_limited_generate_content(prompt)
        log_entry['response'] = response
    except Exception as e:
        log_entry['response'] = f"Error: {str(e)}"
    log_entry['llm_time'] = time.time() - start_time
    
    # Write to log
    log_writer.writerow(log_entry)
    return log_entry

# --- Main Testing Function ---
async def main():
    global request_count, token_count, request_day_count
    
    # Load queries
    queries = load_queries(QUERY_FILE)
    if not queries:
        print("No queries loaded. Exiting.")
        return
    
    # Initialize CSV log
    with open(LOG_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'query', 'iteration', 'embedding_time', 'retrieval_time', 'llm_time', 'response']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each query 4 times
        for query in queries:
            for iteration in range(1, 5):
                print(f"Processing query: {query} (Iteration {iteration})")
                await process_query(query, iteration, writer)
                
                # Reset minute counters if necessary
                if request_count >= RPM_LIMIT or token_count >= TPM_LIMIT:
                    print("Rate limit reached, waiting for next minute...")
                    await asyncio.sleep(PERIOD - (time.time() % PERIOD))
                    request_count = 0
                    token_count = 0

# --- Run the Test ---
if __name__ == "__main__":
    asyncio.run(main())