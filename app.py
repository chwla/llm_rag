import os
import time
import glob
from typing import List, Dict, Any, Tuple, Optional
import json
from datetime import datetime
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from langchain_community.document_loaders import (
    PyMuPDFLoader, 
    TextLoader, 
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
import ollama
from sentence_transformers import CrossEncoder
import numpy as np

# ============================== #
# Configuration
# ============================== #
DOCUMENTS_DIR = "./documents"
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".csv", ".docx", ".doc", ".pptx", ".ppt"]

# Performance optimizations
CHROMA_SETTINGS = {
    "anonymized_telemetry": False,
    "allow_reset": True
}

# ============================== #
# Cached Components (Performance Critical)
# ============================== #
@st.cache_resource
def get_cross_encoder():
    """Cache the cross-encoder model to avoid reloading"""
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@st.cache_resource
def get_vector_collection():
    """Cache vector collection connection"""
    embedding_fn = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest"
    )
    client = chromadb.PersistentClient(
        path="./demo-rag-chroma",
        settings=chromadb.Settings(**CHROMA_SETTINGS)
    )
    return client.get_or_create_collection(
        name="rag_app",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_available_files_and_types():
    """Cache available files and types"""
    try:
        collection = get_vector_collection()
        results = collection.get(limit=1000)
        
        files = set()
        file_types = set()
        
        for metadata in results.get('metadatas', []):
            if metadata:
                if 'source_file' in metadata:
                    files.add(metadata['source_file'])
                if 'file_type' in metadata:
                    file_types.add(metadata['file_type'].upper())
        
        return sorted(list(files)), sorted(list(file_types))
    except:
        return [], []

# ============================== #
# System Prompt (Shortened for Performance)
# ============================== #
system_prompt = """You are a School Safety Management AI assistant specializing in fire safety protocols, disaster management, and training programs for government schools.

## Core Responsibilities
- School fire safety management and protocols
- Emergency response and evacuation procedures  
- Training modules for safety personnel
- Risk assessment and safety equipment guidance
- Compliance with safety regulations

## Response Guidelines
- Prioritize safety-critical information
- Provide step-by-step procedures for emergencies
- Include specific timelines and responsibilities
- Reference relevant safety codes and standards
- Structure responses with clear action items

## Response Format
Use clear, professional language with:
- Immediate Action Items (critical steps)
- Detailed Procedures (step-by-step guidance)
- Compliance Requirements (regulations/standards)
- Follow-up Actions (next steps)

Always prioritize safety and err on the side of caution."""

# ============================== #
# Initialize Session State
# ============================== #
def initialize_session_state():
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'available_documents' not in st.session_state:
        st.session_state.available_documents = []

# ============================== #
# Optimized Document Processing
# ============================== #
def discover_documents() -> List[str]:
    """Discover all supported documents"""
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
        return []
    
    documents = []
    for ext in SUPPORTED_EXTENSIONS:
        pattern = os.path.join(DOCUMENTS_DIR, f"*{ext}")
        documents.extend(glob.glob(pattern))
    
    return sorted(documents)

def get_document_loader(file_path: str, file_type: str):
    """Get appropriate document loader"""
    loaders = {
        'pdf': PyMuPDFLoader,
        'txt': TextLoader,
        'csv': CSVLoader,
        'docx': UnstructuredWordDocumentLoader,
        'doc': UnstructuredWordDocumentLoader,
        'pptx': UnstructuredPowerPointLoader,
        'ppt': UnstructuredPowerPointLoader
    }
    
    loader_class = loaders.get(file_type.lower())
    if not loader_class:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    if file_type.lower() == 'csv':
        return loader_class(file_path, encoding='utf-8')
    else:
        return loader_class(file_path)

def process_backend_document(file_path: str) -> List[Document]:
    """Process document and return chunks"""
    file_name = os.path.basename(file_path)
    file_extension = file_name.split('.')[-1].lower()
    
    try:
        loader = get_document_loader(file_path, file_extension)
        docs = loader.load()
        
        # Add minimal metadata for performance
        file_stats = os.stat(file_path)
        for doc in docs:
            doc.metadata.update({
                'source_file': file_name,
                'file_type': file_extension,
                'file_size': file_stats.st_size
            })
        
        # Optimized chunking settings
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,  # Smaller chunks for faster processing
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "],
        )
        return splitter.split_documents(docs)
    
    except Exception as e:
        st.error(f"Error processing {file_name}: {e}")
        return []

def add_to_vector_collection(all_splits: List[Document], file_name: str):
    """Add document chunks to vector collection"""
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}_{int(time.time())}")

    # Larger batch size for better performance
    BATCH_SIZE = 50
    for i in range(0, len(documents), BATCH_SIZE):
        try:
            collection.upsert(
                documents=documents[i:i+BATCH_SIZE],
                metadatas=metadatas[i:i+BATCH_SIZE],
                ids=ids[i:i+BATCH_SIZE],
            )
            # Reduced sleep time
            time.sleep(0.1)
        except Exception as e:
            st.error(f"Batch {i//BATCH_SIZE + 1} failed: {e}")
            return False
    
    st.session_state.processed_files.add(file_name)
    return True

# ============================== #
# Background Document Loading
# ============================== #
@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_backend_documents():
    """Load documents with caching"""
    document_paths = discover_documents()
    
    if not document_paths:
        return []
    
    try:
        collection = get_vector_collection()
        existing_docs = collection.get(limit=100)  # Reduced limit for speed
        existing_files = set()
        for metadata in existing_docs.get('metadatas', []):
            if metadata and 'source_file' in metadata:
                existing_files.add(metadata['source_file'])
        
        return [os.path.basename(path) for path in document_paths if 
                os.path.basename(path) in existing_files]
    
    except Exception:
        return []

# ============================== #
# Optimized Query and Ranking
# ============================== #
def query_collection(
    prompt: str, 
    n_results: int = 8,  # Reduced from 15
    file_filter: Optional[str] = None,
    file_type_filter: Optional[str] = None
):
    """Query collection with reduced results for speed"""
    collection = get_vector_collection()
    
    where_clause = {}
    if file_filter and file_filter != "All Documents":
        where_clause["source_file"] = file_filter
    if file_type_filter and file_type_filter != "All Types":
        where_clause["file_type"] = file_type_filter.lower()
    
    if where_clause:
        return collection.query(
            query_texts=[prompt], 
            n_results=n_results,
            where=where_clause
        )
    else:
        return collection.query(query_texts=[prompt], n_results=n_results)

def re_rank_cross_encoders(
    prompt: str, 
    documents: List[str], 
    top_k: int = 3
) -> Tuple[str, List[int], List[float]]:
    """Optimized re-ranking with cached model"""
    encoder = get_cross_encoder()  # Use cached model
    
    # Limit document length for faster processing
    truncated_docs = [doc[:1000] for doc in documents]
    pairs = [(prompt, doc) for doc in truncated_docs]
    scores = encoder.predict(pairs)
    
    top_results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    
    top_ids = [i for i, _ in top_results]
    confidence_scores = [float(score) for _, score in top_results]
    top_text = "\n\n".join([documents[i] for i in top_ids])
    
    return top_text, top_ids, confidence_scores

# ============================== #
# Optimized Conversation Management
# ============================== #
def add_to_conversation(question: str, answer: str, confidence: float):
    """Add to conversation with size limit"""
    conversation_entry = {
        'timestamp': datetime.now().isoformat(),
        'question': question,
        'answer': answer[:500] + '...' if len(answer) > 500 else answer,  # Truncate long answers
        'confidence': confidence
    }
    st.session_state.conversation_history.append(conversation_entry)
    
    # Keep only last 5 conversations for better performance
    if len(st.session_state.conversation_history) > 5:
        st.session_state.conversation_history = st.session_state.conversation_history[-5:]

def get_conversation_context() -> str:
    """Get minimal conversation context"""
    if not st.session_state.conversation_history:
        return ""
    
    # Only include last 2 conversations to reduce context size
    context_parts = []
    for entry in st.session_state.conversation_history[-2:]:
        # Truncate for performance
        question = entry['question'][:100]
        answer = entry['answer'][:200]
        context_parts.append(f"Q: {question}\nA: {answer}")
    
    return "\n\n".join(context_parts)

# ============================== #
# Optimized LLM Call
# ============================== #
def call_llm(context: str, prompt: str, conversation_context: str = ""):
    """Optimized LLM call with shorter context"""
    # Truncate context for faster processing
    context = context[:2000] if len(context) > 2000 else context
    conversation_context = conversation_context[:500] if len(conversation_context) > 500 else conversation_context
    
    user_content = f"Context: {context}\n\n"
    
    if conversation_context:
        user_content += f"Previous: {conversation_context}\n\n"
    
    user_content += f"Question: {prompt}"
    
    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        options={
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 500,  # Limit response length for speed
        }
    )
    for chunk in response:
        if not chunk.get("done"):
            yield chunk["message"]["content"]

# ============================== #
# Streamlit App
# ============================== #
if __name__ == "__main__":
    st.set_page_config(
        page_title="School Safety Q&A", 
        page_icon="🏫",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Load documents with caching
    available_docs = load_backend_documents()
    st.session_state.available_documents = available_docs
    
    # Main header
    st.title("🏫 School Safety Management Q&A System")
    st.markdown("*Ask questions about fire safety protocols, disaster management guidelines, and training programs*")
    
    # Quick system check
    try:
        collection = get_vector_collection()
        system_ready = True
    except Exception as e:
        system_ready = False
        st.error("⚠️ System initialization issue. Please ensure Ollama is running.")
        st.code("ollama pull nomic-embed-text\nollama pull llama3.2:3b", language="bash")
    
    # Document status
    if not available_docs and system_ready:
        st.info(f"Add documents to `{DOCUMENTS_DIR}` folder. Supported: PDF, TXT, CSV, DOCX, PPTX")
    elif available_docs:
        st.success(f"📚 **{len(available_docs)} documents loaded** - System ready")
    
    # Simplified filtering (cached)
    available_files, available_types = get_available_files_and_types()
    
    if available_files:
        col1, col2, col3 = st.columns(3)
        with col1:
            file_filter = st.selectbox("Document", ["All Documents"] + available_files[:10])  # Limit options
        with col2:
            type_filter = st.selectbox("Type", ["All Types"] + available_types)
        with col3:
            top_k = st.slider("Results", 1, 5, 3)  # Reduced max for speed
    else:
        file_filter = "All Documents"
        type_filter = "All Types"
        top_k = 3
    
    # Question input
    st.subheader("❓ Ask Your Question")
    prompt = st.text_area(
        "Enter your question:",
        placeholder="What are the key components of a school fire safety plan?",
        height=80
    )
    
    ask = st.button("🚀 Get Answer", type="primary")
    
    if ask and prompt:
        if not available_docs:
            st.warning("⚠️ No documents loaded. Add documents and refresh.")
        else:
            # Performance monitoring
            start_time = time.time()
            
            with st.spinner("🔍 Processing..."):
                # Step 1: Query (optimized)
                results = query_collection(
                    prompt, 
                    n_results=8,  # Reduced
                    file_filter=file_filter if file_filter != "All Documents" else None,
                    file_type_filter=type_filter if type_filter != "All Types" else None
                )
                
                raw_docs = results.get("documents", [[]])[0]
                raw_metadatas = results.get("metadatas", [[]])[0]

                if not raw_docs:
                    st.warning("🔍 No relevant information found. Try rephrasing your question.")
                else:
                    # Step 2: Re-rank (optimized)
                    relevant_text, relevant_ids, confidence_scores = re_rank_cross_encoders(
                        prompt, raw_docs, top_k
                    )
                    
                    # Step 3: Get context (minimal)
                    conversation_context = get_conversation_context()
                    
                    # Step 4: Generate response
                    query_time = time.time() - start_time
                    st.caption(f"Search completed in {query_time:.1f}s")
                    
                    st.markdown("### 📋 Answer:")
                    response_container = st.empty()
                    full_response = ""
                    
                    response_start = time.time()
                    for chunk in call_llm(relevant_text, prompt, conversation_context):
                        full_response += chunk
                        response_container.markdown(full_response + "▌")
                    
                    response_container.markdown(full_response)
                    
                    # Performance metrics
                    total_time = time.time() - start_time
                    generation_time = time.time() - response_start
                    
                    # Results
                    avg_confidence = np.mean(confidence_scores)
                    add_to_conversation(prompt, full_response, avg_confidence)
                    
                    # Show metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        confidence_color = "green" if avg_confidence > 0.7 else "orange" if avg_confidence > 0.4 else "red"
                        st.markdown(f"**Confidence:** <span style='color: {confidence_color}'>{avg_confidence:.0%}</span>", 
                                  unsafe_allow_html=True)
                    
                    # Source information (compact)
                    with st.expander("📄 Sources"):
                        used_files = {raw_metadatas[idx].get('source_file', 'Unknown') 
                                    for idx in relevant_ids if idx < len(raw_metadatas) and raw_metadatas[idx]}
                        if used_files:
                            st.write("• " + " • ".join(sorted(used_files)))

    # Compact conversation history
    if st.session_state.conversation_history:
        with st.expander("💬 Recent Questions"):
            for entry in reversed(st.session_state.conversation_history[-3:]):
                st.write(f"**Q:** {entry['question'][:100]}")
                st.write(f"**A:** {entry['answer'][:150]}...")
                st.write(f"*{entry['confidence']:.0%} confidence*")
                st.divider()
    
    st.markdown("---")
    st.markdown("*School Safety Management Q&A System*")