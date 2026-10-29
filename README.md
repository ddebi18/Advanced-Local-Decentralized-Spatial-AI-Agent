# Edge AI & Local RAG System (Intel Core i3-1215U / 8GB RAM)

This project implements a lightweight, native Python Edge AI agent and Local Retrieval-Augmented Generation (RAG) pipeline tailored for resource-constrained systems without dedicated GPUs.

---

## 1. System Architecture & Memory Budget

| Component | Technology | Memory Footprint (RAM) | Execution Device |
| :--- | :--- | :--- | :--- |
| **OS & Baseline** | Windows 11 / Core Services | ~3.5 GB | CPU |
| **LLM Inference** | `ollama` (`llama3.2:1b`) | ~1.3 GB | CPU / AVX2 |
| **Embedding Model** | `sentence-transformers` (`all-MiniLM-L6-v2`) | ~120 MB | CPU (PyTorch constrained) |
| **Vector Database** | `chromadb` (Persistent, Local SQLite + HNSW) | ~80 MB | Disk / In-Memory cache |
| **Application Runtime** | Native Python | ~100 MB | CPU |
| **Total System RAM** | | **~5.1 GB / 8.0 GB** | Safe headroom: >2.8 GB |

### Key Optimizations:
1. **No Heavy Framework Overhead**: Bypassed LangChain / LlamaIndex in favor of standard Python and the official `ollama` SDK.
2. **PyTorch CPU Limiting**: Constrained to 4 threads (`torch.set_num_threads(4)`) to avoid thread contention across Intel i3 P-cores and E-cores.
3. **KV Cache Control**: Set context window to `num_ctx: 1024` for tool calling and `1536` for RAG, preventing memory bloat.
4. **Targeted Chunking**: Chunks strictly bounded between 300 and 500 characters to keep vector calculations fast and lightweight.
5. **Garbage Collection**: Explicit `gc.collect()` following document ingestion.

---

## 2. Project Structure

```text
├── data.txt          # Sample knowledge base document for Project Falcon
├── requirements.txt  # Minimal dependencies (ollama, chromadb, sentence-transformers)
├── main.py           # Complete end-to-end implementation (Phase 1 & Phase 2)
└── README.md         # Architecture and execution guide
```

---

## 3. Setup & Installation

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Ensure Ollama and Model are Ready
Verify Ollama is running and pull the 1B parameter model:
```bash
ollama serve
ollama pull llama3.2:1b
```

---

## 4. Running the Pipeline

Run the comprehensive test script:
```bash
python main.py
```

### Script Execution Flow:
1. **System Health Check**: Verifies Ollama connectivity and presence of `llama3.2:1b`.
2. **Phase 1 (Tool-Calling Agent Loop)**:
   - Evaluates a time query -> triggers `get_current_time()` tool.
   - Evaluates a math expression -> triggers `calculate()` tool.
   - Evaluates a general query -> responds directly without triggering tools.
3. **Phase 2 (Local Vector DB & RAG Pipeline)**:
   - Ingests `data.txt`, slices it into 300–500 character chunks, and embeds with `all-MiniLM-L6-v2`.
   - Stores embeddings into a persistent ChromaDB index (`./chroma_db`).
   - Retrieves top-2 relevant chunks for factual questions.
   - Injects chunks into a compact prompt and produces grounded answers via `llama3.2:1b`.
