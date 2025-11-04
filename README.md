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

## 🔮 Next Step (Day 3)
We will integrate **LangChain** to build the **Retrieval-Augmented Generation pipeline**, which retrieves semantically relevant documents and uses an **LLM** to generate summarized, cited responses.

