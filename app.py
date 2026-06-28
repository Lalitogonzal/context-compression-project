import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import streamlit as st
import os
import json
import time
import nltk
import ollama
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

# --- Configuration & Styling ---
st.set_page_config(page_title="AI Context Sieve & Audit Engine", page_icon="⚖️", layout="wide")

# Ensure parsing tools are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

EMBEDDING_ENGINE = load_embedding_model()

# --- Pipeline Core Logic Functions ---
def generate_adaptive_chunks(raw_text, chunk_size=3):
    sentences = nltk.sent_tokenize(raw_text)
    chunks = []
    for i in range(len(sentences)):
        chunk = " ".join(sentences[i : i + chunk_size])
        if len(chunk.split()) > 15:
            chunks.append(chunk)
    return chunks

def score_and_retrieve_chunks(chunks, query, top_n=2):
    if not chunks:
        return []
    chunk_embeddings = EMBEDDING_ENGINE.encode(chunks, convert_to_tensor=True)
    query_embedding = EMBEDDING_ENGINE.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, chunk_embeddings)
    scored_chunks = list(zip(cos_scores.tolist(), chunks))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    seen = set()
    unique_top = []
    for score, chunk in scored_chunks:
        if chunk not in seen:
            seen.add(chunk)
            unique_top.append(chunk)
        if len(unique_top) == top_n:
            break
    return unique_top

def build_structured_prompt(context_chunks, query):
    context_str = "\n\n---\n\n".join(context_chunks)
    return f"""You are an advanced data extraction engine. Analyze the provided Text context and extract an objective answer for the Query.

Context:
{context_str}

Query: {query}

CRITICAL INSTRUCTION: You must respond ONLY with a valid JSON object. Do not include markdown formatting, backticks, wrapping, conversational filler, or introductory phrases.

Desired JSON Format:
{{
  "query_answered": true/false,
  "confidence_score": 0.0 to 1.0,
  "extracted_answer": "your answer string here",
  "supporting_quotes": ["quote 1 from text"]
}}"""

# ==========================================
# EVALUATION & METRICS ENGINE LAYER
# ==========================================
def evaluate_run_performance(orig_words, comp_words, parse_success, confidence, query_answered):
    # Pillar 2: Context Compression calculation
    tokens_saved_pct = (1.0 - (comp_words / orig_words)) * 100 if orig_words > 0 else 0
    
    # Pillar 1: Accuracy Proxy Heuristic Score
    # A run is high quality if it parses perfectly, finds the answer, and has strong confidence
    if parse_success and query_answered:
        quality_score = confidence * 100
    elif parse_success and not query_answered:
        quality_score = 40.0  # Context parsed but clean data was missing
    else:
        quality_score = 0.0   # Structural crash
        
    return {
        "compression_ratio_pct": round(tokens_saved_pct, 2),
        "system_quality_score": round(quality_score, 2),
        "status": "PASS ✅" if parse_success and query_answered else "WARN ⚠️" if parse_success else "FAIL ❌"
    }

# --- Streamlit Presentation Layer UI ---
st.title("⚖️ Production Pipeline Dashboard & Evaluation Auditor")
st.markdown("Ingest massive datasets, trigger parallel evaluation queries, and monitor raw system health performance.")

with st.sidebar:
    st.header("1. Data Ingestion")
    uploaded_file = st.file_uploader("Upload a context document (.txt)", type=["txt"])
    
    st.header("2. Compression Hyperparameters")
    chunk_window = st.slider("Sentence Window Size", min_value=1, max_value=5, value=3)
    top_n_chunks = st.slider("Max Chunks to Retain", min_value=1, max_value=5, value=2)

st.header("3. Parallel Target Query Input")
queries_input = st.text_area(
    "Enter one target query per line for simultaneous execution processing:",
    value="What is the main conclusion of this text?\nWhat dataset or prior modeling limitations are discussed?"
)

queries_list = [q.strip() for q in queries_input.split("\n") if q.strip()]

if st.button("🚀 Run Live End-to-End Pipeline Evaluation", type="primary"):
    if not uploaded_file:
        st.error("Please upload a source file in the left sidebar to proceed.")
    elif not queries_list:
        st.error("Please enter at least one question to analyze.")
    else:
        raw_document = uploaded_file.read().decode("utf-8")
        original_word_count = len(raw_document.split())
        
        # Performance logging trackers for data visualization later
        query_names = []
        savings_data = []
        quality_data = []
        
        with st.status("Executing active system lifecycle blocks...", expanded=True) as status:
            st.write("🔄 Stage 1 & 2: Tokenizing and forming adaptive text windows...")
            chunks = generate_adaptive_chunks(raw_document, chunk_window)
            
            st.write("⚡ Stage 3 to 7: Vector scoring and local model inference...")
            tabs = st.tabs([f"Query {i+1}" for i in range(len(queries_list))])
            
            for index, query in enumerate(queries_list):
                with tabs[index]:
                    st.markdown(f"#### **Query:** *\"{query}\"*")
                    
                    # Track localized latency
                    start_time = time.time()
                    
                    # Execute Pipeline Loop
                    top_chunks = score_and_retrieve_chunks(chunks, query, top_n_chunks)
                    final_prompt = build_structured_prompt(top_chunks, query)
                    
                    compressed_word_count = len(" ".join(top_chunks).split())
                    
                    response = ollama.chat(
                        model='llama3.2',
                        messages=[{'role': 'user', 'content': final_prompt}],
                        format='json'
                    )
                    latency = time.time() - start_time
                    raw_output = response['message']['content'].strip()
                    
                    # Parse & Clean Output
                    parse_success = False
                    confidence = 0.0
                    query_answered = False
                    json_data = {}
                    
                    try:
                        json_data = json.loads(raw_output)
                        parse_success = True
                        confidence = float(json_data.get("confidence_score", 0.5))
                        query_answered = bool(json_data.get("query_answered", False))
                    except Exception:
                        parse_success = False
                    
                    # Run Evaluation Calculations Live!
                    eval_results = evaluate_run_performance(
                        original_word_count, compressed_word_count, 
                        parse_success, confidence, query_answered
                    )
                    
                    # Store data for our graphing step
                    query_names.append(f"Q{index+1}")
                    savings_data.append(eval_results["compression_ratio_pct"])
                    quality_data.append(eval_results["system_quality_score"])
                    
                    # --- Render Visual Metrics Column Blocks ---
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Context Saved", f"{eval_results['compression_ratio_pct']}%", help="Pillar 2: Words stripped away")
                    col2.metric("Output Quality Score", f"{eval_results['system_quality_score']}/100", help="Pillar 1: Extracted value strength")
                    col3.metric("Schema Status", eval_results["status"], help="Pillar 3: Pure JSON structural compliance validation")
                    col4.metric("Inference Speed", f"{latency:.2f}s", help="Time elapsed during local processing")
                    
                    # Render final answer data cards
                    if parse_success:
                        st.markdown(f"**Extracted System Answer:**\n> {json_data.get('extracted_answer', 'None')}")
                        with st.expander("Show Retained Text Fragments Sent to LLM"):
                            st.write(top_chunks)
                    else:
                        st.error("JSON parsing crash! The local model broken strict constraint formats.")
                        st.code(raw_output)
                        
            status.update(label="Complete Evaluation System Pass Accomplished!", state="complete")
            
        # ==========================================
        # GRAPH VISUALIZATION LAYER
        # ==========================================
        st.header("📊 Multi-Query Engine Audit Analysis")
        
        fig, ax1 = plt.subplots(figsize=(7, 3), dpi=150)
        plt.style.use('fast')
        
        # Plot savings line chart
        color = '#1f77b4'
        ax1.set_xlabel('Simultaneous Executed Queries')
        ax1.set_ylabel('Token Compute Saved (%)', color=color)
        bars = ax1.bar(query_names, savings_data, color=color, alpha=0.6, width=0.4, label='Context Savings %')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 100)
        
        # Mirror axes for precision accuracy quality metric line overlays
        ax2 = ax1.twinx()  
        color = '#ff7f0e'
        ax2.set_ylabel('Engine Integrity Score (0-100)', color=color)
