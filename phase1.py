import os
import json
import nltk
import ollama
from sentence_transformers import SentenceTransformer, util

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# Ensure parsing tools are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# Define globally accessible, modular engines
EMBEDDING_ENGINE = SentenceTransformer('all-MiniLM-L6-v2')

# ==========================================
# 1 & 2. SENTENCE SPLITTER & CHUNK GENERATOR
# ==========================================
def generate_adaptive_chunks(raw_text, chunk_size=3):
    sentences = nltk.sent_tokenize(raw_text)
    chunks = []
    for i in range(len(sentences)):
        chunk = " ".join(sentences[i : i + chunk_size])
        if len(chunk.split()) > 15: # Filter noise
            chunks.append(chunk)
    return chunks

# ==========================================
# 3 & 4. EMBEDDING ENGINE & RETRIEVAL SCORER
# ==========================================
def score_and_retrieve_chunks(chunks, query, top_n=2):
    if not chunks:
        return []
    
    chunk_embeddings = EMBEDDING_ENGINE.encode(chunks, convert_to_tensor=True)
    query_embedding = EMBEDDING_ENGINE.encode(query, convert_to_tensor=True)
    
    cos_scores = util.cos_sim(query_embedding, chunk_embeddings)
    scored_chunks = list(zip(cos_scores.tolist(), chunks))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    # Return unique top scoring chunks
    seen = set()
    unique_top = []
    for score, chunk in scored_chunks:
        if chunk not in seen:
            seen.add(chunk)
            unique_top.append(chunk)
        if len(unique_top) == top_n:
            break
    return unique_top

# ==========================================
# 5. PROMPT BUILDER
# ==========================================
def build_structured_prompt(context_chunks, query):
    context_str = "\n\n---\n\n".join(context_chunks)
    
    prompt = f"""You are an advanced data extraction engine. Analyze the provided Text context and extract an objective answer for the Query.

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
    return prompt

# ==========================================
# 6 & 7. OLLAMA & JSON KNOWLEDGE EXTRACTOR
# ==========================================
def execute_pipeline(file_path, user_query):
    # Load raw document
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_document = f.read()

    print("[1/5] Running Sentence Splitter & Chunking Engine...")
    chunks = generate_adaptive_chunks(raw_document)
    
    print("[2/5] Scoring Vector Embeddings...")
    top_chunks = score_and_retrieve_chunks(chunks, user_query)
    
    print("[3/5] Injecting Context into Prompt Builder...")
    final_prompt = build_structured_prompt(top_chunks, user_query)
    
    print("[4/5] Executing Inference via Ollama (Forcing JSON mode)...")
    # CRITICAL FIX: We tell Ollama explicitly to restrict the model to JSON structures
    response = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': final_prompt}],
        format='json'  
    )
    raw_output = response['message']['content'].strip()
    
    print("[5/5] Processing JSON Knowledge Extractor...")
    try:
        # A much more aggressive cleaning approach to strip away markdown code blocks if they exist
        cleaned_output = raw_output
        if "```json" in cleaned_output:
            cleaned_output = cleaned_output.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned_output:
            cleaned_output = cleaned_output.split("```")[1].split("```")[0].strip()
            
        json_data = json.loads(cleaned_output)
        return json_data
    except Exception as e:
        return {
            "parsing_failed": True,
            "raw_output_received": raw_output,
            "error_msg": str(e)
        }

# Run the complete integrated system
if __name__ == "__main__":
    print("=== PIPELINE ACTIVATED ===")
    query = "What is the main conclusion of this text?"
    
    structured_data = execute_pipeline("document.txt", query)
    
    print("\n=== FINAL PIPELINE JSON OUTPUT ===")
    print(json.dumps(structured_data, indent=2))
    print("===================================\n")
