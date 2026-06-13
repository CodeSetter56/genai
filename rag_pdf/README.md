# RAG PDF Chatbot

A multi-PDF question-answering chatbot built from scratch using LangChain, Chroma, Google Gemini embeddings, and Groq LLM. Ask questions across multiple documents and get accurate, context-grounded answers.

---

## Features

- Multi-PDF support with intelligent routing
- Incremental PDF indexing — add or remove PDFs without re-indexing
- Query decomposition for multi-part questions
- Conversation memory with history limiting
- Query rewriting for follow-up questions
- Structured metadata descriptions for accurate routing
- Streaming responses
- Rate-limit-safe batched embedding

---

## Tech Stack

| Component | Tool |
|---|---|
| Embeddings | Google Gemini (`gemini-embedding-001`) |
| Vector Store | Chroma (local) |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Framework | LangChain |
| Package Manager | uv |

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/CodeSetter56/genai.git
cd genai/rag_pdf
```

**2. Install dependencies**
```bash
uv sync
```

**3. Add API keys**

Create a `.env` file in `rag_pdf/`:
```bash
GOOGLE_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
```

Get your keys:
- Gemini: https://aistudio.google.com
- Groq: https://console.groq.com

**4. Add your PDFs**

Drop any PDF files into the `data/` folder. The folder is created automatically on first run if it doesn't exist.

**5. Run**
```bash
uv run main.py
```

On first run, the app indexes all PDFs and saves the vector store to `chroma_db/`. Subsequent runs load the existing DB instantly and automatically detect any added or removed PDFs.

---

## Project Structure

```
rag_pdf/
├── main.py          # entry point, orchestrates the full RAG pipeline
├── embeddings.py    # embedding model setup
├── ingestion.py     # PDF loading and chunking
├── vector_db.py     # Chroma vector store creation, loading, and updates
├── llm.py           # LLM setup and answer generation
├── memory.py        # chat history, query rewriting, and query decomposition
├── router.py        # PDF description generation and query routing
├── indexer.py       # incremental index state — tracks indexed/removed PDFs
├── data/            # put your PDFs here (gitignored)
└── chroma_db/       # auto-generated vector store (gitignored)
```

---

## RAG Pipeline

```
PDF files → load → chunk → describe → embed → store (first run only)
                                                     ↓
                         sync: detect new/removed PDFs → update index
                                                     ↓
query → decompose → rewrite → route → retrieve → generate → answer
```

---

## Techniques Used

### 1. Retrieval-Augmented Generation (RAG)

The core architecture. Retrieved chunks from the documents are passed as context to the LLM, grounding answers in your actual files rather than the model's training data.

```python
# main.py
sub_results = vector_store.similarity_search_with_score(rewritten, k=5, filter=...)
response = answer_query(results, query, chat_history)
```

### 2. Chunking with Overlap

PDFs are split into overlapping chunks so context isn't lost at boundaries. `RecursiveCharacterTextSplitter` splits on natural boundaries like paragraphs before resorting to hard cuts.

```python
# ingestion.py
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=300)
chunks = text_splitter.split_documents(documents)
```

The high overlap (300 of 700 characters) is intentional — short factual chunks like a joining date sentence need enough surrounding context to score well in similarity search.

### 3. Semantic Embeddings

Chunks are converted into vectors using Google's Gemini embedding model. Similarity search finds chunks whose *meaning* is closest to the query, not just keyword matches.

```python
# embeddings.py
def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
```

### 4. Persistent Vector Store & Incremental Indexing

Chroma persists embeddings to disk so PDFs are only re-indexed on first run. On subsequent runs, `indexer.py` compares `data/` against `indexed_files.json` to detect changes — new PDFs are chunked and added, removed PDFs are deleted by metadata filter. No full rebuild needed.

```python
# indexer.py
# adding new PDFs
new_documents, _ = load_all_pdfs(data_dir, only_files=new_pdfs)
vector_store.add_documents(new_chunks)

# removing deleted PDFs
vector_store.delete(where={"source": os.path.join(data_dir, pdf_file)})
```

The `indexed_files.json` fallback derives from `pdf_descriptions.json` if missing, so existing setups upgrade gracefully without re-indexing.

### 5. Batched Embedding with Rate Limit Handling

Gemini's free tier caps at 100 embed requests per minute. Chunks are embedded in batches of 50 with a 60-second pause between batches to stay within limits.

```python
# vector_db.py
for i in range(0, len(chunks), batch_size):
    batch_chunks = chunks[i:i + batch_size]
    # ... embed batch
    if i + batch_size < len(chunks):
        time.sleep(60)  # avoid rate limit
```

### 6. Structured PDF Descriptions for Routing

Instead of a vague one-sentence summary, each PDF gets a structured metadata description generated on first index. The router uses the `Contains` field to judge relevance precisely, distinguishing documents like a joining letter vs an offer letter that would otherwise look similar.

```python
# router.py
prompt = f"""Return a structured description in exactly this format:
Type: <document type>
Topic: <main subject>
Entities: <key people, companies, locations mentioned>
Contains: <comma separated list of specific information>

Document excerpt: {sample_text}"""
```

### 7. Multi-PDF Routing with Metadata Filtering

The router selects only the relevant PDFs for each sub-query using the structured descriptions. Retrieval is then filtered to only those PDFs, preventing unrelated documents from polluting the context.

```python
# router.py
router_prompt = f"""...
Rules:
- Use the 'Contains' field to judge relevance
- Return ONLY strictly necessary filenames, comma separated"""

# main.py
sub_results = vector_store.similarity_search_with_score(
    rewritten, k=5,
    filter={"source": {"$in": [os.path.join(DATA_DIR, f) for f in selected_files]}}
)
```

### 8. Query Decomposition

Multi-part questions are split into individual sub-queries before retrieval. Each sub-query is routed and retrieved independently, then results are merged and deduplicated before going to the LLM. This prevents one part of the question from drowning out the other in similarity search.

```python
# memory.py
def decompose_query(query, get_model):
    # returns a list of standalone sub-queries, or [query] if single-part
    sub_queries = [q.strip() for q in response.split("\n") if q.strip()]
    return sub_queries

# main.py — deduplicate across sub-query results
for doc, score in sub_results:
    chunk_id = doc.page_content[:100]
    if chunk_id not in seen_ids:
        seen_ids.add(chunk_id)
        results.append((doc, score))
```

### 9. Query Rewriting

Each sub-query is rewritten using chat history to resolve pronouns and references before retrieval. The prompt is deliberately conservative — it only resolves explicit references and never infers topic or adds facts from history unprompted.

```python
# memory.py
rewrite_prompt = f"""...
Rules:
- Only resolve clear pronouns like "it", "that", "they"
- Do not infer topic or context from history if the query stands on its own
- Do not add answers, facts, or company names from history unless explicitly referenced
- When in doubt, keep the query as close to the original wording as possible"""
```

### 10. Conversation Memory with History Limiting

Chat history is stored and injected into the LLM prompt for conversational context. History is capped at the last 7 turns to prevent token overflow on long conversations.

```python
# memory.py
CHAT_HISTORY_LIMIT = 7

def format_chat_history(chat_history):
    recent = chat_history[-CHAT_HISTORY_LIMIT:]
    ...
```

### 11. LangChain Chains & Streaming Responses

The prompt and LLM are composed into a chain using LangChain's pipe operator. Answers are streamed token by token so responses appear immediately rather than waiting for full generation.

```python
# llm.py
chain = prompt | get_model()
for chunk in chain.stream({"context": context, "question": query, "chat_history": history_str}):
    print(chunk.content, end="", flush=True)
```

### 12. Lazy Initialization

The LLM is not instantiated at import time — only on the first call to `get_model()`, after `load_dotenv()` has run. This prevents a `GroqError` caused by importing the module before the API key is available in the environment.

```python
# llm.py
_model = None
def get_model():
    global _model
    if _model is None:
        _model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=1000)
    return _model
```

### 13. Temperature = 0 & Prompt Engineering

Temperature 0 makes the LLM deterministic — critical for a factual Q&A bot. The system prompt instructs the model to use documents as primary truth, supplement with general knowledge only when needed, and never hallucinate specific numbers or facts.

```python
# llm.py
_model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=1000)

prompt = ChatPromptTemplate.from_template("""
    Use the provided context as your PRIMARY source of truth.
    You may supplement with general knowledge ONLY when the document provides partial information.
    When you use general knowledge, clearly mention it.
    Do NOT hallucinate specific numbers or facts not present in either the context or your training.
    If you truly cannot answer, say "I don't know based on the provided document."
    ...""")
```

---

## How to Add or Remove PDFs

**Add:** Drop the new PDF into `data/` and run `uv run main.py`. The app detects it, chunks and embeds only that file, and adds it to the existing vector store.

**Remove:** Delete the PDF from `data/` and run `uv run main.py`. The app detects it's missing, deletes its chunks from Chroma by metadata filter, and removes it from the index.

No manual re-indexing or deletion of `chroma_db/` needed in either case.

---