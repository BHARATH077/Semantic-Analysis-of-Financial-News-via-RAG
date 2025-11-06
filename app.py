import streamlit as st
from transformers import pipeline
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# Load model & data
df = pd.read_csv("financial_documents_with_embeddings.csv")
index = faiss.read_index("financial_news_index.faiss")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
llm = pipeline("text2text-generation", model="google/flan-t5-small")

# Custom retriever
class FAISS_Retriever:
    def __init__(self, index, df, embedding_model, k=3):
        self.index = index
        self.df = df
        self.embedding_model = embedding_model
        self.k = k
    
    def retrieve(self, query):
        query_vector = self.embedding_model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_vector, self.k)
        results = []
        for idx in indices[0]:
            results.append({
                "title": self.df.iloc[idx]["title"],
                "text": self.df.iloc[idx]["text"],
                "source": self.df.iloc[idx]["source"],
                "link": self.df.iloc[idx].get("link", None)
            })
        return results

retriever = FAISS_Retriever(index, df, embedding_model, k=3)

# Entity extraction
import spacy
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ["ORG","MONEY","PERCENT","DATE"]:
            entities.append({"text": ent.text, "label": ent.label_})
    return entities

def extract_relationships(text):
    doc = nlp(text)
    relationships = []
    for sent in doc.sents:
        orgs = [ent.text for ent in sent.ents if ent.label_=="ORG"]
        moneys = [ent.text for ent in sent.ents if ent.label_=="MONEY"]
        percents = [ent.text for ent in sent.ents if ent.label_=="PERCENT"]
        if orgs and (moneys or percents):
            for org in orgs:
                for value in moneys + percents:
                    relationships.append({"company": org, "value": value, "sentence": sent.text})
    return relationships

# RAG + Citations
def format_context_with_citations(retrieved_docs):
    formatted = ""
    for i, doc in enumerate(retrieved_docs, 1):
        citation = f"{doc['title']}" if doc['link'] is None else f"{doc['title']} ({doc['link']})"
        formatted += f"{i}. {doc['text']} [{citation}]\n\n"
    return formatted

def generate_answer_with_entities(query):
    docs = retriever.retrieve(query)
    context = format_context_with_citations(docs)
    prompt = f"Answer the following question using the context below and include citations.\n\nContext:\n{context}\n\nQuestion: {query}"
    answer = llm(prompt, max_length=250)[0]['generated_text']
    
    # Extract entities & relationships
    entities = []
    relationships = []
    for doc in docs:
        entities.extend(extract_entities(doc['text']))
        relationships.extend(extract_relationships(doc['text']))
    
    return answer, entities, relationships

# Streamlit UI
st.title("Financial News Semantic Analysis (RAG)")
query = st.text_input("Enter your financial query:")

if query:
    answer, entities, relationships = generate_answer_with_entities(query)
    
    st.subheader("Generated Answer")
    st.write(answer)
    
    st.subheader("Extracted Entities")
    st.json(entities)
    
    st.subheader("Extracted Relationships")
    st.json(relationships)
