# Statistics RAG Assistant

A Streamlit application that answers questions from `STATS.pdf` using FastEmbed
embeddings, an in-memory Qdrant vector store, and NVIDIA's Nemotron model.

## Setup

1. Create and activate a Python 3.11 environment.
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and set `NVIDIA_API_KEY`.
4. Run the app:
   ```powershell
   streamlit run main.py
   ```

## Environment

Set `NVIDIA_API_KEY` in a local `.env` file or in the Streamlit secrets
configuration. Never commit API keys.
