# RAG PDF Chatbot

A multi-PDF question-answering chatbot built from scratch using LangChain, Chroma, Google Gemini embeddings, and Groq LLM. Ask questions across multiple documents and get accurate, context-grounded answers.

---

## Features

- Multi-PDF support with intelligent routing
- Conversation memory across questions
- Query rewriting for follow-up questions
- Streaming responses
- Auto-generated document descriptions
- Rate-limit-safe batched embedding
- **Incremental PDF Indexing**: Efficiently updates the vector store when PDFs are added or removed, avoiding full re-indexing.
- **Query Decomposition**: Breaks down complex, multi-part questions into simpler sub-queries for more targeted retrieval.
- **Chat History Limiting**: Maintains a concise conversational context by limiting the chat history size.

---

## Tech Stack

| Component       | Tool                                   |
| --------------- | -------------------------------------- |
| Embeddings      | Google Gemini (`gemini-embedding-001`) |
| Vector Store    | Chroma (local)                         |
| LLM             | Groq (`llama-3.3-70b-versatile`)       |
| Framework       | LangChain                              |
| Package Manager | uv                                     |

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

On first run, the app indexes all PDFs and saves the vector store to `chroma_db/`. Subsequent runs load the existing DB instantly and check for new or removed PDFs to update the index incrementally.

---

## Project Structure

```
rag_pdf/
├── main.py          # entry point, orchestrates the full RAG pipeline
├── embeddings.py    # embedding model setup
├── ingestion.py     # PDF loading and chunking
├── vector_db.py     # Chroma vector store creation and loading
├── llm.py           # LLM setup and answer generation
├── memory.py        # chat history formatting, query rewriting, and decomposition
├── router.py        # PDF routing and description generation
├── data/            # put your PDFs here (gitignored)
└── chroma_db/       # auto-generated vector store (gitignored)
```

---

## RAG Pipeline

The full pipeline runs in sequence on every query:

```
PDF files → load → chunk → embed → store (first run or new/removed PDFs)
                                        ↓
query → decompose → rewrite → route → retrieve → generate → answer
```

---

## Techniques Used

### 1. Retrieval-Augmented Generation (RAG)

The core architecture. Instead of relying on the LLM's training data, we retrieve relevant chunks from the document and pass them as context. This grounds the answers in your actual documents.

```python
# main.py
results = vector_store.similarity_search_with_score(rewritten, k=7, filter=...)
response = answer_query(results, query, chat_history)
```

### 2. Chunking with Overlap

Large PDFs are split into overlapping chunks so context isn't lost at boundaries. `RecursiveCharacterTextSplitter` tries to split on natural boundaries like paragraphs before resorting to hard cuts.

```python
# ingestion.py
text_splitter = RecursiveCharacterTextSplitter(chunk_size=550, chunk_overlap=150)
chunks = text_splitter.split_documents(documents)
```

### 3. Semantic Embeddings

Text chunks are converted into vector representations using Google's Gemini embedding model. Similarity search then finds chunks whose meaning is closest to the query — not just keyword matches.

```python
# embeddings.py
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
```

### 4. Persistent Vector Store & Incremental Indexing

Chroma stores embeddings to disk so PDFs are only re-indexed when the `chroma_db/` folder is missing. Subsequent runs load instantly. Additionally, the system intelligently detects new or removed PDFs and updates the index incrementally without rebuilding the entire database.

```python
# main.py (simplified for illustration)
# Removal of PDFs
removed_pdfs = get_removed_pdfs(DATA_DIR, indexed_files)
if removed_pdfs:
    pdf_descriptions, indexed_files = remove_pdfs_from_index(
        vector_store, DATA_DIR, removed_pdfs, pdf_descriptions, indexed_files
    )

# Addition of new PDFs
new_pdfs = get_new_pdfs(DATA_DIR, indexed_files)
if new_pdfs:
    new_chunks = load_new_pdf_chunks(DATA_DIR, new_pdfs, CHUNK_SIZE, CHUNK_OVERLAP)
    vector_store = add_to_db(new_chunks, embedding_model)

# indexer.py (snippet for deletion)
# vector_store.delete(where={"source": os.path.join(data_dir, pdf_file)})
```

### 5. Batched Embedding with Rate Limit Handling

Gemini's free tier allows 100 embed requests per minute. Large PDFs are embedded in batches of 50 with a 60-second pause between batches to stay within limits.

```python
# vector_db.py
for i in range(0, len(chunks), batch_size):
    batch_chunks = chunks[i:i + batch_size]
    # ... embed batch
    if i + batch_size < len(chunks):
        time.sleep(60)  # avoid rate limit
```

### 6. LangChain Chains

The prompt and LLM are composed into a chain using LangChain's pipe operator. This makes the flow declarative and easy to extend.

```python
# llm.py
chain = prompt | get_model()
for chunk in chain.stream({"context": context, "question": query, "chat_history": history_str}):
    print(chunk.content, end="", flush=True)
```

### 7. Streaming Responses

Instead of waiting for the full response, answers are streamed token by token as they're generated. On slower models or longer answers this feels significantly faster.

```python
# llm.py
for chunk in chain.stream({"context": context, "question": query, "chat_history": history_str}):
    content = str(chunk.content)
    print(content, end="", flush=True)
    full_response += content
```

### 8. Conversation Memory & Limiting

Each question and answer is stored in `chat_history` and injected into the prompt. The history is limited to the most recent turns to maintain focus and manage token usage effectively.

```python
# memory.py
CHAT_HISTORY_LIMIT = 7

def format_chat_history(chat_history):
    recent = chat_history[-CHAT_HISTORY_LIMIT:]
    # ... format history ...

# main.py
chat_history = []
response = answer_query(results, query, chat_history)
chat_history.append({"human": query, "assistant": response})
```

### 9. Query Rewriting

Follow-up questions like "and what about Cognizant?" are rewritten into standalone queries like "what is my joining date at Cognizant" before hitting the vector store. This improves retrieval quality for conversational queries.

```python
# memory.py
def rewrite_query(query, chat_history, get_model):
    if not chat_history:
        return query
    recent = chat_history[-CHAT_HISTORY_LIMIT:]
    history_str = format_chat_history(recent)
    rewrite_prompt = f"""Given this conversation history: {history_str}
    Rewrite this follow-up question as a standalone search query: "{query}"
    Return only the rewritten query, nothing else."""
    response = get_model().invoke(rewrite_prompt)
    return str(response.content)
```

### 10. Query Decomposition

Complex user queries are decomposed into a list of simpler, standalone sub-queries. Each sub-query is then processed independently through the RAG pipeline, allowing for more comprehensive information retrieval.

```python
# memory.py
def decompose_query(query, get_model):
    prompt = f"""Analyze this question and determine if it contains multiple distinct information requests.
            If it does, split it into separate standalone search queries.
            If it is a single request, return it as-is.

            Question: "{query}"

            Rules:
            - Only split if there are genuinely separate pieces of information being requested
            - Each sub-query should be fully standalone
            - Return one query per line, nothing else"""

    response = get_model().invoke(prompt)
    sub_queries = [q.strip() for q in str(response.content).strip().split("\n") if q.strip()]
    return sub_queries

# main.py (usage)
sub_queries = decompose_query(query, get_model)
```

### 11. Multi-PDF Routing with Auto-Generated Descriptions & Metadata Filtering

When multiple PDFs are loaded, the router uses an LLM to decide which documents are relevant to the query. Descriptions are auto-generated from each PDF's content on first run and saved to disk — so the router works for any documents, not just hardcoded ones. Retrieval is further refined by filtering the similarity search results to only include chunks from the router-selected PDFs, preventing irrelevant documents from polluting the context.

```python
# router.py (description generation)
def generate_pdf_description(filename, chunks):
    sample_text = "\n".join([c.page_content for c in chunks[:3]])
    prompt = f"In one sentence, describe what this document is about: {sample_text}"
    response = get_model().invoke(prompt)
    return str(response.content)

# main.py (routing and filtering)
selected_files = route_query(rewritten, pdf_descriptions)
sub_results = vector_store.similarity_search_with_score(
    rewritten, k=5,
    filter={"source": {"$in": [os.path.join(DATA_DIR, f) for f in selected_files]}} # type: ignore
)
```

Example — asking "what is my Cognizant salary" routes only to `cognizant.pdf`, while "compare TCS and Cognizant salaries" routes to both offer letter PDFs.

### 12. Lazy Initialization

The LLM is not instantiated at import time. It's created on the first call to `get_model()`, after `load_dotenv()` has run and the API key is available. This prevents a `GroqError` that would otherwise occur because the module is imported before the environment is loaded.

```python
# llm.py
_model = None

def get_model():
    global _model
    if _model is None:
        _model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=1000)
    return _model
```

### 13. Temperature = 0

Setting temperature to 0 makes the LLM deterministic — it always picks the most probable token rather than sampling randomly. This is critical for a factual document Q&A bot where consistency and accuracy matter more than creativity.

```python
# llm.py
_model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=1000)
```

### 14. Prompt Engineering

The system prompt explicitly instructs the LLM to use documents as primary truth, allow general knowledge only to fill gaps, and clearly label when it does so. This reduces hallucination while keeping the bot useful for questions the document only partially answers.

```python
# llm.py
prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant.
    Use the provided context as your PRIMARY source of truth.
    You may supplement with your general knowledge ONLY when the document provides
    partial information and general knowledge can reasonably fill the gap.
    When you use general knowledge, clearly mention it.
    Do NOT hallucinate specific numbers or facts not present in either the context or your training.
    If you truly cannot answer, say "I don't know based on the provided document."
    ...
""")
```

---

## How to Add New PDFs

1.  Drop the new PDF into `data/`
2.  Run `uv run main.py`
    The app will automatically detect new PDFs, load, chunk, embed, and generate a description for them, adding them to the existing vector store without requiring a full rebuild.

---

## Known Limitations

- Gemini free tier: 100 embed requests/min — large PDFs take a few minutes to index
- No web interface — terminal only
