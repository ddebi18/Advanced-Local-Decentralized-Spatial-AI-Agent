"""
Edge AI & Local RAG System for Resource-Constrained Hardware
Target Hardware: Intel Core i3-1215U (6 cores / 8 threads), 8GB System RAM, No dGPU.
Model: llama3.2:1b (Ollama)
Embeddings: all-MiniLM-L6-v2 (Sentence-Transformers, CPU)
Vector Store: ChromaDB (Local Persistent)

Author: Edge AI Midterm Project Implementation
"""

import os
import re
import gc
import sys
import json
import math
import datetime
from typing import List, Dict, Any, Optional

# Suppress Windows symlink warning from huggingface_hub on non-developer mode systems
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Third-party imports
import psutil
import torch
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import ollama

# =====================================================================
# HARDWARE OPTIMIZATIONS FOR INTEL i3-1215U & 8GB RAM
# =====================================================================
# Constrain PyTorch thread pool to avoid thread thrashing and memory spikes
torch.set_num_threads(4)
DEVICE = "cpu"

# Constant configurations
MODEL_NAME = "llama3.2:1b"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
DATA_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.txt")
COLLECTION_NAME = "edge_ai_knowledge_base"


# =====================================================================
# GLOBAL CACHED EMBEDDING MODEL (Single instance in RAM ~120MB)
# =====================================================================
_embedding_model: Optional[SentenceTransformer] = None

def get_embedding_model() -> SentenceTransformer:
    """
    Lazy-load and cache the lightweight sentence-transformer model on CPU.
    Keeps memory footprint strictly under ~120MB.
    """
    global _embedding_model
    if _embedding_model is None:
        print(f"[*] Loading embedding model '{EMBEDDING_MODEL_NAME}' onto CPU...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
        print("[+] Embedding model loaded successfully.")
    return _embedding_model


# =====================================================================
# PHASE 1: TOOL DEFINITIONS & TOOL-CALLING AGENT LOOP
# =====================================================================

def get_current_time() -> str:
    """
    Get the current local date, time, and timezone.
    Returns:
        str: Formatted current timestamp string.
    """
    now = datetime.datetime.now()
    return now.strftime("Current Date and Time: %Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    """
    Safely evaluate a basic mathematical expression.
    Supported operations: +, -, *, /, %, **, sqrt, abs, round.
    
    Args:
        expression: A mathematical expression string, e.g., '144 / 12' or '2 ** 8'.
    Returns:
        str: The calculated result or an error message.
    """
    # Whitelist allowed characters for safety (no arbitrary code execution)
    clean_expr = expression.strip()
    if not re.match(r"^[\d\s\+\-\*\/\%\(\)\.\,eE]|sqrt|abs|round|pi$", clean_expr):
        return f"Error: Expression contains unsupported characters or operators."

    # Safe mathematical namespace
    safe_dict = {
        "__builtins__": {},
        "sqrt": math.sqrt,
        "abs": abs,
        "round": round,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        result = eval(clean_expr, safe_dict, {})
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero is undefined."
    except Exception as exc:
        return f"Error evaluating expression: {str(exc)}"


def get_system_status() -> str:
    """
    Get current host system resource metrics including CPU usage percent,
    CPU core count, RAM utilization (used/total/percent), and system disk usage.
    
    Returns:
        str: Human-readable system performance and status string.
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=0.5)
        cpu_cores = psutil.cpu_count(logical=True)
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / (1024 ** 3)
        ram_total_gb = ram.total / (1024 ** 3)
        ram_percent = ram.percent

        system_drive = os.path.splitdrive(os.path.abspath("."))[0] + "\\"
        disk = psutil.disk_usage(system_drive)
        disk_percent = disk.percent

        return (
            f"CPU Usage: {cpu_percent}% ({cpu_cores} cores) | "
            f"RAM: {ram_used_gb:.2f} GB / {ram_total_gb:.2f} GB ({ram_percent}%) | "
            f"Disk ({system_drive}): {disk_percent}% used"
        )
    except Exception as exc:
        return f"Error retrieving system status: {str(exc)}"


def search_local_database(query: str, top_k: int = 2) -> str:
    """
    Search the local ChromaDB vector database knowledge base for relevant document chunks.
    
    Args:
        query: The search question or keywords to look up in the documentation.
        top_k: Number of relevant document chunks to retrieve (default is 2).
        
    Returns:
        str: Retrieved document text chunks separated by dividers, or a not found message.
    """
    try:
        chunks = retrieve_context(query=query, top_k=top_k)
        if not chunks:
            return "No relevant documents found."
        return "\n\n---\n\n".join(chunks)
    except Exception as exc:
        return f"Error searching local database: {str(exc)}"


# Map of callable tool names for execution
AVAILABLE_TOOLS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "get_system_status": get_system_status,
    "search_local_database": search_local_database,
}

# Tool schemas compatible with Ollama function calling
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current local date and time. Use this when the user asks for the current time, today's date, or day.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Performs mathematical calculations. Use this whenever math, arithmetic, or numerical computation is required.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. '125 * 8' or 'sqrt(144)'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_status",
            "description": (
                "Returns current host hardware and system resource metrics including CPU usage percent, "
                "CPU core count, RAM memory used and total in GB, RAM percent, and system disk usage percent. "
                "Use this tool whenever the user asks about CPU, RAM, memory usage, disk space, "
                "system performance, or computer health."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_local_database",
            "description": (
                "Searches the local ChromaDB vector database and stored Project Falcon knowledge base documents. "
                "Use this tool whenever the user asks questions about Project Falcon, memory allocation limits, "
                "hardware architecture, maintenance schedule, contact info, network communication protocols, "
                "or any stored project documentation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or keywords to find in the stored documents, e.g. 'memory allocation limit'",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "The number of document chunks to retrieve (default is 2)",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def run_tool_calling_agent(user_prompt: str) -> str:
    """
    Executes a native Python tool-calling loop using Ollama and llama3.2:1b.
    Detects if the model requests a function call, executes it locally,
    feeds the result back to the model, and outputs the final response.
    """
    print(f"\n[Agent] User query: '{user_prompt}'")

    messages = [
        {
            "role": "system",
            "content": (
                "You are an efficient AI assistant running locally on edge hardware. "
                "You have access to tools for getting the current time, evaluating math calculations, "
                "checking system hardware status (CPU/RAM/disk), and searching the local database knowledge base. "
                "CRITICAL RULES:\n"
                "1. Only call tools when the user's question explicitly requires them (e.g., only call get_current_time if the user asks for the current time/date; call get_system_status for CPU/RAM/disk metrics; call search_local_database for stored documents/Project Falcon).\n"
                "2. If no tool is required, answer directly without invoking tools.\n"
                "3. When a tool returns a result, you MUST use that result to provide the final answer. Never say your knowledge cutoff prevents you from answering if a tool output was provided."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    try:
        # Step 1: Initial call to llama3.2:1b with tool definitions
        # Keeping num_ctx low (1024) drastically reduces KV cache RAM usage
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            options={
                "temperature": 0.0,  # Deterministic tool selection
                "num_ctx": 1024,      # Low memory footprint
            },
        )

        message = response.get("message", {})
        tool_calls = message.get("tool_calls", [])

        # Step 2: Check if model invoked any tool
        if tool_calls:
            print(f"[Agent] Model triggered {len(tool_calls)} tool call(s).")
            # Append model's assistant message with tool calls
            messages.append(message)

            for tool_call in tool_calls:
                function_info = tool_call.get("function", {})
                func_name = function_info.get("name")
                raw_args = function_info.get("arguments", {})

                # Parse arguments if returned as string
                if isinstance(raw_args, str):
                    try:
                        func_args = json.loads(raw_args)
                    except Exception:
                        func_args = {}
                else:
                    func_args = raw_args or {}

                # Guard: llama3.2:1b occasionally echoes the full tool-schema object
                # (with keys 'type', 'function', 'parameters') instead of just the
                # parameter values. Unwrap the nested 'parameters' dict if detected.
                if isinstance(func_args, dict) and "parameters" in func_args and "type" in func_args:
                    func_args = func_args.get("parameters") or {}

                # Strip None values so optional params fall back to Python defaults
                func_args = {k: v for k, v in func_args.items() if v is not None}

                print(f"[Agent] Executing tool: '{func_name}' with args: {func_args}")

                if func_name in AVAILABLE_TOOLS:
                    try:
                        tool_result = AVAILABLE_TOOLS[func_name](**func_args)
                    except Exception as e:
                        tool_result = f"Error executing {func_name}: {e}"
                else:
                    tool_result = f"Error: Tool '{func_name}' is not recognized."

                print(f"[Agent] Tool Output: {tool_result}")

                # Step 3: Append tool output back to conversation history
                messages.append({
                    "role": "tool",
                    "content": f"Tool '{func_name}' result: {tool_result}",
                })

            # Explicitly instruct 1B model to formulate final answer using the tool output
            messages.append({
                "role": "user",
                "content": f"Using the tool result ('{tool_result}'), provide a direct, concise answer to the original question: '{user_prompt}'",
            })

            # Step 4: Final call to generate answer with tool results in context
            final_response = ollama.chat(
                model=MODEL_NAME,
                messages=messages,
                options={
                    "temperature": 0.2,
                    "num_ctx": 1024,
                },
            )
            final_content = final_response.get("message", {}).get("content", "").strip()
            return final_content
        else:
            # Model responded directly without needing tools
            return message.get("content", "").strip()

    except Exception as err:
        return f"[Agent Error] Failed during tool-calling loop: {err}"


# =====================================================================
# PHASE 2: LOCAL VECTOR DB & RAG PIPELINE
# =====================================================================

def init_chroma_db(persist_path: str = CHROMA_PERSIST_DIR) -> chromadb.PersistentClient:
    """
    Initializes a persistent ChromaDB client with telemetry disabled
    to minimize network and background thread overhead on edge devices.
    """
    os.makedirs(persist_path, exist_ok=True)
    client = chromadb.PersistentClient(
        path=persist_path,
        settings=Settings(anonymized_telemetry=False)
    )
    return client


def split_text_into_chunks(text: str, min_chars: int = 300, max_chars: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits text into coherent chunks between 300 and 500 characters.
    Preserves whole paragraphs and sentences without slicing words or creating orphan punctuation.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current_chunk = ""

    for para in paragraphs:
        # Standardize whitespace within paragraph
        normalized_para = " ".join(line.strip() for line in para.split("\n") if line.strip())

        if not current_chunk:
            if len(normalized_para) <= max_chars:
                current_chunk = normalized_para
            else:
                # If paragraph itself is longer than max_chars, split along sentences
                sentences = re.findall(r'[^.!?]+[.!?]?', normalized_para)
                sub_chunk = ""
                for s in sentences:
                    s = s.strip()
                    if not s:
                        continue
                    if len(sub_chunk) + len(s) + 1 <= max_chars:
                        sub_chunk = (sub_chunk + " " + s).strip()
                    else:
                        if len(sub_chunk) >= min_chars:
                            chunks.append(sub_chunk)
                            sub_chunk = s
                        else:
                            sub_chunk = (sub_chunk + " " + s).strip()
                            if len(sub_chunk) >= max_chars:
                                chunks.append(sub_chunk)
                                sub_chunk = ""
                if sub_chunk:
                    chunks.append(sub_chunk)
        else:
            if len(current_chunk) + len(normalized_para) + 2 <= max_chars:
                current_chunk = f"{current_chunk}\n\n{normalized_para}"
            else:
                chunks.append(current_chunk)
                current_chunk = normalized_para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def ingest_document_to_chroma(
    file_path: str = DATA_FILE_PATH,
    collection_name: str = COLLECTION_NAME,
    force_refresh: bool = False
) -> int:
    """
    Ingests data.txt into ChromaDB:
    1. Reads file.
    2. Splits into 300-500 char chunks.
    3. Computes embeddings using all-MiniLM-L6-v2 on CPU.
    4. Stores chunks in persistent ChromaDB collection.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    client = init_chroma_db()
    
    # Use get_or_create_collection for robust cross-version compatibility
    collection = client.get_or_create_collection(name=collection_name)
    
    # Check if collection already exists to avoid redundant re-embedding on startup
    if not force_refresh and collection.count() > 0:
        count = collection.count()
        print(f"[+] Using existing ChromaDB collection '{collection_name}' with {count} chunks.")
        return count

    # If force_refresh and collection has data, clean existing items
    existing = collection.get()
    if existing and existing.get("ids"):
        collection.delete(ids=existing["ids"])

    print(f"[*] Reading and chunking document: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = split_text_into_chunks(content, min_chars=300, max_chars=500, overlap=50)
    print(f"[*] Generated {len(chunks)} chunks (each 300-500 characters).")

    if not chunks:
        print("[!] Warning: No chunks generated from document.")
        return 0

    # Embed chunks in CPU-friendly batches
    model = get_embedding_model()
    print("[*] Generating embeddings on CPU...")
    embeddings = model.encode(chunks, batch_size=8, show_progress_bar=False, device=DEVICE)

    # Store into ChromaDB
    ids = [f"chunk_{i:04d}" for i in range(len(chunks))]
    metadatas = [{"source": os.path.basename(file_path), "char_len": len(c)} for c in chunks]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print(f"[+] Successfully stored {len(chunks)} chunks in ChromaDB collection '{collection_name}'.")

    # Clean up memory immediately
    del embeddings
    gc.collect()

    return len(chunks)


def retrieve_context(query: str, top_k: int = 2) -> List[str]:
    """
    Embeds the user query, searches ChromaDB, and returns the top 2 most relevant chunks.
    """
    client = init_chroma_db()
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    model = get_embedding_model()
    query_embedding = model.encode([query], device=DEVICE).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    documents = results.get("documents", [[]])[0]
    return documents


def run_rag_pipeline(user_query: str) -> str:
    """
    RAG Execution Pipeline:
    1. Embed query and retrieve top 2 relevant chunks from ChromaDB.
    2. Format a compact, context-grounded system prompt.
    3. Pass prompt to llama3.2:1b via Ollama.
    4. Return concise, grounded response.
    """
    print(f"\n[RAG] User Query: '{user_query}'")
    print("[RAG] Retrieving top 2 relevant chunks from ChromaDB...")

    try:
        retrieved_chunks = retrieve_context(user_query, top_k=2)
    except Exception as e:
        return f"[RAG Error] ChromaDB retrieval failed: {e}"

    if not retrieved_chunks:
        return "No relevant context found in local database."

    print(f"[RAG] Retrieved {len(retrieved_chunks)} chunk(s):")
    for idx, doc in enumerate(retrieved_chunks, 1):
        preview = doc.replace("\n", " ")[:120]
        print(f"   [{idx}] ({len(doc)} chars): {preview}...")

    # Compact context injection to minimize Ollama context buffer
    context_str = "\n\n".join([f"--- Context Chunk {i+1} ---\n{doc}" for i, doc in enumerate(retrieved_chunks)])

    system_prompt = (
        "You are a helpful edge AI assistant. Answer the user question based on the provided context facts. "
        "Be concise and factual. If the context does not contain the answer, state that the documentation does not contain this information."
    )

    user_prompt = f"Context Information:\n{context_str}\n\nQuestion: {user_query}\n\nAnswer:"

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": 0.1,  # Low temperature for factual grounding
                "num_ctx": 1536,      # Enough for ~1000 char context + prompt + answer
            },
        )
        answer = response.get("message", {}).get("content", "").strip()
        return answer
    except Exception as err:
        return f"[RAG Error] Ollama inference failed: {err}"


# =====================================================================
# MAIN TEST & DEMO DRIVER
# =====================================================================

def verify_ollama_status() -> bool:
    """Verifies that the Ollama daemon is running and llama3.2:1b is available."""
    try:
        models = ollama.list()
        model_list = [m.model for m in models.models] if hasattr(models, "models") else []
        # Check model name loosely (e.g. 'llama3.2:1b' or 'llama3.2:1b-instruct-q4_K_M')
        matches = [m for m in model_list if "llama3.2:1b" in m]
        if not matches:
            print(f"[!] Warning: '{MODEL_NAME}' not found in `ollama list`. Available: {model_list}")
            print(f"    Please run: ollama pull {MODEL_NAME}")
            return False
        return True
    except Exception as e:
        print(f"[X] Cannot connect to Ollama daemon: {e}")
        print("    Please ensure Ollama is running (`ollama serve`).")
        return False


def main():
    print("=" * 70)
    print("  EDGE AI & RAG SYSTEM (Intel i3-1215U / 8GB RAM Optimized)")
    print("=" * 70)

    # 1. Health check
    if not verify_ollama_status():
        sys.exit(1)

    # -------------------------------------------------------------
    # PHASE 1 DEMONSTRATION: TOOL-CALLING AGENT LOOP
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print(" >>> PHASE 1: TOOL-CALLING AGENT LOOP")
    print("=" * 50)

    test_queries_tools = [
        "What is the current time and date right now?",
        "Can you calculate (45 * 18) + (144 / 12)?",
        "Who is the president of the moon?",  # Direct response without tool call
    ]

    for query in test_queries_tools:
        result = run_tool_calling_agent(query)
        print(f"[Agent Response]:\n{result}\n")
        print("-" * 50)

    # -------------------------------------------------------------
    # PHASE 2 DEMONSTRATION: LOCAL CHROMA VECTOR DB & RAG PIPELINE
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print(" >>> PHASE 2: INGESTION & LOCAL RAG PIPELINE")
    print("=" * 50)

    # 2.1 Ingest sample data.txt
    ingest_document_to_chunks = ingest_document_to_chroma(
        file_path=DATA_FILE_PATH,
        collection_name=COLLECTION_NAME,
        force_refresh=True,  # Fresh build for demonstration
    )

    # 2.2 RAG Queries against data.txt
    rag_test_queries = [
        "What is the memory allocation limit for the ChromaDB and all-MiniLM-L6-v2 embedding model?",
        "Who is the technical lead for hardware maintenance and what is their email?",
        "What network port and protocol does Falcon use for emergency fallback polling?",
        "What is the recipe for chocolate chip cookies?",  # Out of domain query
    ]

    for q in rag_test_queries:
        rag_answer = run_rag_pipeline(q)
        print(f"[RAG Final Answer]:\n{rag_answer}\n")
        print("-" * 50)

    print("\n[+] All Phase 1 and Phase 2 demonstrations finished successfully.")


if __name__ == "__main__":
    main()
