# context-compression-project

# Adaptive Context Compression Engine for Local LLMs

A production-grade, modular RAG (Retrieval-Augmented Generation) preprocessing pipeline built to dramatically slash LLM API token consumption, reduce context window clutter, and optimize local inference performance.

## 🏗️ System Architecture

Our engine processes massive unstructured inputs through a multi-stage token-saving framework:

```text
  [Raw Document] (.txt)
        │
        ▼
 [Sentence Splitter] (NLTK Tokenizer)
        │
        ▼
 [Adaptive Chunking Engine] (Sliding Context Windows)
        │
        ▼
 [Vector Embedding Model] (all-MiniLM-L6-v2)
        │
        ▼
 [Retrieval Scorer & Filter] (Cosine Similarity Ranking)
        │
        ▼
 [Dynamic Prompt Builder] (Structural Constraint Framing)
        │
        ▼
 [Local Inference Engine] (Ollama / Llama 3.2)
        │
        ▼
 [JSON Knowledge Extractor] (Structural Output Validator)
```

## 📈 Key Performance Metrics & Core Benefits
* **Cost Efficiency:** Reduces token input by **95%+** per query, preserving API budget or local compute resources.
* **Cohesive Chunking:** Moves past brittle string splits (`split('.')`) to embrace NLTK tokenization with sliding context windowing, ensuring target concepts maintain linguistic completeness.
* **Deterministic Structural Delivery:** Forces erratic local conversational LLMs into strict, programmatic validation via native JSON schemas.

## 🛠️ Stack & Technologies
* **Language:** Python 3.13
* **Local Compute Server:** Ollama (Hosting `Llama 3.2`)
* **Vector Models:** HuggingFace `sentence-transformers`
* **Natural Language Processor:** NLTK (Natural Language Toolkit)

## 🏃‍♂️ How to Run locally

1. **Clone & Spin up Virtual Environment:**
   ```bash
   git clone <your-repo-link>
   cd context-compression-project
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install ollama sentence-transformers nltk
   ```

2. **Ensure Ollama App is running and Model is pulled:**
   ```bash
   ollama pull llama3.2
   ```

3. **Run Pipeline Execution:**
   ```bash
   python phase1.py
   ```
