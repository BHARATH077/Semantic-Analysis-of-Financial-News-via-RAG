# Semantic-Analysis-of-Financial-News-via-RAG

# Step 1 — Data Collection & Preprocessing

## 🎯 Objective
Prepare the text corpus for the RAG system by collecting, cleaning, and structuring financial news and SEC filings.

---

## 🧩 Steps Completed
1. Installed dependencies: `pandas`, `langchain`, `sentence-transformers`, `faiss-cpu`.
2. Collected financial news from **Yahoo Finance RSS** feeds.
3. Loaded and simulated **SEC filings** for major tech companies.
4. Cleaned and merged multiple text sources into a unified corpus.
5. Saved the preprocessed data as `financial_documents.csv`.

---

## 📁 Output Files
- `financial_documents.csv` — Unified dataset combining financial news and filings.

---

# Step 2 — Semantic Embedding & Vector Indexing (FAISS)

## 🎯 Objective
Convert financial text data into semantic embeddings using **Sentence Transformers**, and store them in a **FAISS** index for similarity-based retrieval.

---

## 🧩 Steps Completed
1. Loaded the preprocessed dataset from Day 1 (`financial_documents.csv`).
2. Used `sentence-transformers/all-MiniLM-L6-v2` to embed each text entry.
3. Initialized a **FAISS** index for storing embeddings and enabling fast vector search.
4. Validated the setup by testing retrieval with example financial queries.
5. Saved the FAISS index and data mappings for downstream RAG operations.

---

## 📊 Example Query
**Query:** “Apple quarterly earnings report”

**Results:**
1. *Apple reports Q2 revenue of $117B…*  
2. *AAPL stock sees uptick after earnings release…*  
3. *Apple’s service revenue drives growth.*

---

## 📁 Output Files
- `financial_news_index.faiss` — FAISS vector index storing all embeddings  
- `financial_documents_with_embeddings.csv` — Documents with metadata  

---

## 🧠 Concepts Learned
- Semantic embeddings with Sentence Transformers  
- Vector-based similarity search using FAISS  
- Building a scalable base for RAG systems  

---

# Step 3 — Retrieval-Augmented Generation (RAG) Pipeline

## 🎯 Objective
Build a **RAG system** that combines **semantic retrieval** (FAISS) with a **language model** to answer queries using real financial documents.

---

## 🧩 Steps Completed
1. Loaded FAISS index and preprocessed financial document metadata.
2. Built a custom **FAISS retriever** class:
   - Retrieves the top-k semantically similar documents for a query.
3. Integrated a **local LLM** (`google/flan-t5-small`) for summarization and question answering.
4. Tested the RAG pipeline with example queries:
   - Query: “Apple quarterly revenue performance”
   - Output: Contextualized answer based on retrieved news + filings.
   
---

## 📁 Key Concepts Learned
- Retrieval-Augmented Generation (RAG) workflow
- Connecting **vector embeddings** with LLMs
- Combining multiple sources for context-aware responses
- Context concatenation and prompting for summarization

---

# Step 4 — Semantic Query Answering with Citations

## 🎯 Objective
Enhance the RAG pipeline to:
- Provide context-aware answers
- Include **citations** to the source documents
- Ensure answers are concise and verifiable

---

## 🧩 Steps Completed
1. Defined `format_context_with_citations()` to concatenate retrieved documents with source info.
2. Modified RAG function (`generate_answer_with_citations`) to:
   - Include citations in the answer
   - Use top-k documents from the FAISS retriever
   - Generate human-readable summaries with LLM
3. Tested the pipeline with queries such as:
   - “Microsoft cloud revenue growth in 2023”
   - “Apple quarterly revenue performance”
4. Added a utility to display **cited sources** alongside generated answers.

---

## 📁 Key Concepts Learned
- RAG with citations
- Enhancing transparency in LLM outputs
- Merging semantic retrieval with source attribution
- Preparing the system for auditable financial analysis

---

# Step 5 — Entity Extraction & Relationship Modeling

## 🎯 Objective
Extract structured **financial entities and relationships** from retrieved documents to augment the RAG system for more precise, auditable answers.

---

## 🧩 Steps Completed
1. Installed and configured **spaCy** for Named Entity Recognition.
2. Defined `extract_entities()` to capture:
   - Companies (ORG)
   - Financial metrics (MONEY, PERCENT)
   - Dates (DATE)
3. Defined `extract_relationships()` to identify **Company -> Value** relationships in sentences.
4. Integrated entity and relationship extraction with the RAG pipeline:
   - Now, generated answers include:
     - Contextual answer
     - List of extracted entities
     - List of relationships

---

## 🔍 Example Output
**Query:** “Apple quarterly revenue”  

**Answer:** Context-aware summary with citations  

**Entities Extracted:**
- Apple Inc (ORG)
- $117B (MONEY)
- Q2 (DATE)

**Relationships Extracted:**
- Apple Inc -> $117B: Apple reported Q2 revenue of $117B…

---

## 📁 Key Concepts Learned
- Named Entity Recognition (NER) for financial texts
- Simple relationship extraction heuristics
- Enhancing RAG systems with structured knowledge
- Preparing for user-facing query augmentation

---

# Day 6 — Streamlit UI & Deployment

## 🎯 Objective
Deploy the RAG system with a **user-friendly Streamlit interface** for semantic financial query answering.

---

## 🧩 Steps Completed
1. Installed **Streamlit** for interactive web UI.
2. Integrated the full **RAG pipeline**:
   - FAISS retrieval of top-k documents
   - LLM-based context-aware answer generation
   - Inclusion of **citations**
   - Entity extraction and relationship modeling
3. Built a **web interface** (`app.py`) with:
   - Text input for queries
   - Display of generated answer
   - JSON view of extracted entities and relationships
4. Ran Streamlit in Colab using **ngrok** for public access.
5. Tested end-to-end workflow for multiple financial queries.

---

## 📁 Key Files for GitHub
- `financial_documents_with_embeddings.csv` — Preprocessed text corpus
- `financial_news_index.faiss` — Vector index for retrieval
- `app.py` — Streamlit web interface
- **All Day 1–5 code** consolidated into `financial_rag_pipeline.ipynb` (optional)
- README files for each day

---

## 🔮 Outcome
- Fully **deployable RAG system** for financial text analysis
- Supports:
  - Semantic query answering
  - Context citations
  - Entity extraction
  - Relationship modeling
- Enables scalable, auditable insights for financial analysis

---






