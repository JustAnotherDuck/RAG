import os
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document
from sentence_transformers import SentenceTransformer # For final chunk embedding
import chromadb # For vector storage
import uuid # For generating unique IDs for ChromaDB

# --- Configuration ---
TEXT_FILE_PATH = "extracted_llamaparse.txt"
EMBEDDING_MODEL_NAME_SEMANTIC_SPLIT = "sentence-transformers/all-MiniLM-L6-v2" # For LlamaIndex splitter
EMBEDDING_MODEL_NAME_FINAL_CHUNKS = "sentence-transformers/all-MiniLM-L6-v2" # For final RAG embeddings
CHROMA_DB_PATH = "./chroma_db_semantic" # Path to store ChromaDB data
CHROMA_COLLECTION_NAME = "semantic_rag_collection"

# --- 1. Load Your Text ---
try:
    with open(TEXT_FILE_PATH, "r", encoding="utf-8") as f:
        text_content = f.read()
    print(f"Successfully loaded text from {TEXT_FILE_PATH}")
except FileNotFoundError:
    print(f"Error: File not found at {TEXT_FILE_PATH}")
    exit()

document = Document(text=text_content)

# --- 2. Perform Semantic Chunking (using LlamaIndex) ---
print("Initializing semantic splitter and embedding model for chunking...")
# Embedding model for the semantic splitter
embed_model_splitter = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME_SEMANTIC_SPLIT)

# Initialize the SemanticSplitterNodeParser
# You might need to install sentence-splitter: pip install sentence-splitter
splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95, # Lower for more chunks, higher for fewer
    embed_model=embed_model_splitter
)

print("Performing semantic chunking...")
nodes = splitter.get_nodes_from_documents([document])
final_semantic_chunks_text = [node.get_content() for node in nodes]

if not final_semantic_chunks_text:
    print("No semantic chunks were generated. Check your text content or splitter settings.")
    exit()

print(f"Generated {len(final_semantic_chunks_text)} semantic chunks.")
# for i, chunk_text in enumerate(final_semantic_chunks_text):
# print(f"--- Semantic Chunk {i+1} ---")
# print(chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text) # Print snippet
# print("\n")

# --- 3. Embed the Final Semantic Chunks ---
print(f"Initializing embedding model '{EMBEDDING_MODEL_NAME_FINAL_CHUNKS}' for final chunks...")
final_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME_FINAL_CHUNKS)

print("Embedding final semantic chunks...")
final_chunk_embeddings = final_embedding_model.encode(final_semantic_chunks_text, show_progress_bar=True)
print(f"Successfully embedded {len(final_chunk_embeddings)} chunks.")

# --- 4. Store Chunks and Their Embeddings in ChromaDB ---
print(f"Setting up ChromaDB at path: {CHROMA_DB_PATH}")
# Persistent client
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
# Or in-memory client: client = chromadb.Client()

# Get or create the collection
# Note: ChromaDB uses its own embedding function by default if not specified during collection creation
# or if embeddings are not directly provided during add.
# Since we are providing our own embeddings, this is fine.
print(f"Getting or creating ChromaDB collection: {CHROMA_COLLECTION_NAME}")
collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

# Prepare data for ChromaDB
# ChromaDB needs IDs for each document.
chunk_ids = [str(uuid.uuid4()) for _ in final_semantic_chunks_text]
embeddings_list = [emb.tolist() for emb in final_chunk_embeddings] # Convert numpy arrays to lists

print("Adding documents and embeddings to ChromaDB collection...")
collection.add(
    documents=final_semantic_chunks_text,
    embeddings=embeddings_list, # Provide the pre-computed embeddings
    ids=chunk_ids
)

print(f"Successfully added {collection.count()} items to the ChromaDB collection '{CHROMA_COLLECTION_NAME}'.")
print("Process complete! Your semantic chunks and their embeddings are stored.")

# --- To verify, you can query (example) ---
if collection.count() > 0:
     query_text = "What is the main topic of the document?" # Example query
     query_embedding = final_embedding_model.encode([query_text])[0].tolist()

     results = collection.query(
         query_embeddings=[query_embedding],
         n_results=2 # Get top 2 results
     )
     print("\n--- Example Query Results ---")
     print(f"Query: {query_text}")
     if results and results.get('documents'):
         for i, doc in enumerate(results['documents'][0]):
             print(f"Result {i+1}:")
             print(doc[:300] + "...") # Print snippet of retrieved chunk
             print(f"Distance: {results['distances'][0][i] if results.get('distances') else 'N/A'}")
     else:
         print("No results found for the example query.")