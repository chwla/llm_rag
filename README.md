# SET UP

python3 -m venv venv

source venv/bin/activate

ollama pull nomic-embed-text

ollama pull llama3.2:3b

pip install chromadb sentence-transformers streamlit pymupdf langchain-community

streamlit run app.py
