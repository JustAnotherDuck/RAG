from google import genai
import chromadb
from sentence_transformers import SentenceTransformer
import requests


# --- Configuration ---
QUERY = "What is the duration of the sounds samples collection and the number of subjects samples were collected from?"  # Example user query
CHROMA_DB_PATH = "chroma_db_semantic"
CHROMA_COLLECTION_NAME = "semantic_rag_collection_bge_v1.5"  # ChromaDB collection name
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
TOP_K = 5
client = genai.Client(api_key="AIzaSyADiMx-RWGuuFT5amWb1VsZAfVO3kJysQk")


# --- Load Embedding Model ---

print("Loading embedding model for query...")
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# --- Embed User Query ---
print("Embedding query...")
query_embedding = embed_model.encode([QUERY])[0].tolist()

# --- Query ChromaDB ---
print("Querying ChromaDB for similar chunks...")
ch_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = ch_client.get_collection(name=CHROMA_COLLECTION_NAME)

results = collection.query(query_embeddings=[query_embedding], n_results=TOP_K)
retrieved_chunks = results['documents'][0]

# --- Construct Prompt for LLM ---
context = "\n\n".join(retrieved_chunks)
prompt = f"""You are a helpful assistant. Use the following context to answer the user's question.

Context:
{context}

Question:
{QUERY}

Answer:"""

# --- Query Gemini LLM ---

print ("Sending prompt to Gemini LLM...")
response = client.models.generate_content(
  model="gemini-2.0-flash", contents=prompt,
)
print(response.text)
#print(context)
