import os
import uuid
import chromadb
from pathlib import Path
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import SentenceTransformer

# --- Configuration ---
TEXT_FILE_PATH = "extracted_llamaparse.txt"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
CHROMA_DB_PATH = "./chroma_db_semantic"
CHROMA_COLLECTION_NAME = "test_chunking"

# --- Custom Embedding Wrapper for BGE ---
# --- Load Text ---
try:
    with open(TEXT_FILE_PATH, "r", encoding="utf-8") as f:
        text_content = f.read()
    print(f" Loaded text from {TEXT_FILE_PATH}")
except FileNotFoundError:
    print(f" File not found at {TEXT_FILE_PATH}")
    exit()

document = Document(text=text_content)

# --- Semantic Chunking ---
print(" Initializing BGE embedding model for semantic chunking...")
embed_model = HuggingFaceEmbedding(EMBEDDING_MODEL_NAME)

print(" Performing semantic chunking...")
splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=embed_model
)
nodes = splitter.get_nodes_from_documents([document])
final_chunks = [node.get_content() for node in nodes]

if not final_chunks:
    print(" No semantic chunks generated.")
    exit()

stringchunks = "\n\n".join(final_chunks)
# Save chunks to a text file
text_file = Path/"chunks_text.txt"
with open(text_file, 'w', encoding='utf-8') as f:
    f.write(stringchunks)


print(f" Generated {len(final_chunks)} semantic chunks.")

# --- Final Embeddings ---
print(f" Loading BGE model '{EMBEDDING_MODEL_NAME}' for final chunk embeddings...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print(" Embedding chunks...")
chunk_embeddings = embedding_model.encode(final_chunks, show_progress_bar=True)

# --- Store in ChromaDB ---
print(f" Setting up ChromaDB at path: {CHROMA_DB_PATH}")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

chunk_ids = [str(uuid.uuid4()) for _ in final_chunks]
embeddings_list = [emb.tolist() for emb in chunk_embeddings]

print(" Adding chunks + embeddings to ChromaDB...")
collection.add(
    documents=final_chunks,
    embeddings=embeddings_list,
    ids=chunk_ids
)

print(f"Stored {collection.count()} chunks in collection '{CHROMA_COLLECTION_NAME}'.")

# --- Optional: Test Retrieval ---
print("\n Running test query...")
test_query = "What is the main topic of the document?"
query_embedding = embedding_model.encode(["query: " + test_query])[0].tolist()

results = collection.query(query_embeddings=[query_embedding], n_results=2)

print("\n--- Example Query Results ---")
print(f"Query: {test_query}")
if results and results.get('documents'):
    for i, doc in enumerate(results['documents'][0]):
        print(f"\nResult {i+1}:")
        print(doc[:300] + "...")
        print(f"Distance: {results['distances'][0][i]:.4f}")
else:
    print("No results found.")


