import os
import time
import glob
from typing import List, Dict, Any, Tuple, Optional
import json
from datetime import datetime
from functools import lru_cache
import asyncio

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

# FAQ Data
FAQ_DATA = [
    {
        "question": "What is the National School Safety Policy?",
        "answer": "The National School Safety Policy is a set of guidelines issued by the National Disaster Management Authority (NDMA), Government of India, aimed at ensuring the safety and security of children, teachers, and staff in schools against natural and man-made disasters. It covers structural and non-structural safety, capacity building, and disaster preparedness in all schools across India."
    },
    {
        "question": "What are the key components of fire safety in schools?",
        "answer": "Key components include: 1) Proper fire detection and alarm systems, 2) Clearly marked evacuation routes, 3) Regular fire drills (at least quarterly), 4) Fire extinguishers placed at strategic locations, 5) Staff training on fire safety protocols, and 6) Electrical safety measures to prevent fire hazards."
    },
    {
        "question": "How often should fire drills be conducted in schools?",
        "answer": "As per the National School Safety Guidelines, fire drills should be conducted at least once every quarter (3 months). However, it's recommended to conduct them monthly for the first year in new schools or after significant staff changes, then transition to quarterly drills."
    },
    {
        "question": "What is the recommended teacher-to-student ratio during emergencies?",
        "answer": "During emergencies, the recommended ratio is 1 teacher for every 10 students in primary schools and 1 teacher for every 15 students in secondary schools. This ensures proper supervision during evacuation procedures."
    },
    {
        "question": "What types of disasters should school safety plans address?",
        "answer": "School safety plans should address: 1) Fires, 2) Earthquakes, 3) Floods, 4) Cyclones, 5) Chemical spills/industrial accidents, 6) Terrorism threats, 7) Medical emergencies, and 8) Pandemics/health crises."
    }
]

# ============================== #
# Cached Components
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

@st.cache_data(ttl=300)
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
# Navbar Function
# ============================== #
def create_navbar():
    """Create a custom navbar using Streamlit columns"""
    # Custom CSS for navbar styling
    st.markdown("""
    <style>
    .navbar {
        background: linear-gradient(90deg, #1f4e79 0%, #2e6da4 100%);
        padding: 1rem 0;
        margin-bottom: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        width: 100%;
    }
    .navbar-title {
        color: white;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .navbar-subtitle {
        color: #b8d4f0;
        font-size: 0.9rem;
        margin: 0;
        margin-top: 5px;
    }
    .nav-tab {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 0 0.5rem;
        font-weight: 500;
        white-space: nowrap;
    }
    .nav-tab:hover {
        background: rgba(255,255,255,0.2);
        transform: translateY(-2px);
    }
    .nav-tab.active {
        background: white;
        color: #1f4e79;
        font-weight: bold;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .nav-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0 2rem;
        width: 100%;
        box-sizing: border-box;
    }
    .nav-tabs-container {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    [data-testid="stSidebar"] {
        margin-top: -85px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create navbar container
    st.markdown('<div class="navbar"><div class="nav-container">', unsafe_allow_html=True)
    
    # Left side - Title and subtitle
    st.markdown("""
    <div>
        <div class="navbar-title">
            🏫 School Safety Management System
        </div>
        <div class="navbar-subtitle">
            AI-Powered Safety Assistant for Educational Institutions
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Right side - Navigation tabs
    st.markdown('<div class="nav-tabs-container">', unsafe_allow_html=True)
    
    if st.button("🤖 AI Assistant", key="nav_ai", help="Chat with AI Assistant"):
        st.session_state.active_tab = "AI Assistant"
    
    if st.button("❓ FAQ", key="nav_faq", help="Frequently Asked Questions"):
        st.session_state.active_tab = "FAQ"
    
    st.markdown('</div></div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ============================== #
# System Prompt
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
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "AI Assistant"
    if 'messages' not in st.session_state:
        st.session_state.messages = []

# ============================== #
# Document Processing Functions
# ============================== #
def discover_documents() -> List[str]:
    """Discover all supported documents in the documents directory"""
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
        return []

    documents = []
    for ext in SUPPORTED_EXTENSIONS:
        pattern = os.path.join(DOCUMENTS_DIR, f"*{ext}")
        documents.extend(glob.glob(pattern))

    return sorted(documents)

def get_document_loader(file_path: str, file_type: str):
    """Get appropriate document loader based on file type"""
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
    """Process document from backend directory and return chunks"""
    file_name = os.path.basename(file_path)
    file_extension = file_name.split('.')[-1].lower()

    try:
        loader = get_document_loader(file_path, file_extension)
        docs = loader.load()

        # Add file metadata
        file_stats = os.stat(file_path)
        for doc in docs:
            doc.metadata.update({
                'source_file': file_name,
                'file_type': file_extension,
                'file_path': file_path,
                'file_size': file_stats.st_size,
                'modified_time': datetime.fromtimestamp(file_stats.st_mtime).isoformat()
            })

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
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

    BATCH_SIZE = 32
    for i in range(0, len(documents), BATCH_SIZE):
        try:
            collection.upsert(
                documents=documents[i:i+BATCH_SIZE],
                metadatas=metadatas[i:i+BATCH_SIZE],
                ids=ids[i:i+BATCH_SIZE],
            )
            time.sleep(0.5)
        except Exception as e:
            st.error(f"Batch {i//BATCH_SIZE + 1} failed: {e}")
            return False

    st.session_state.processed_files.add(file_name)
    return True

def load_backend_documents():
    """Load all documents from the backend directory into vector store"""
    if st.session_state.documents_loaded:
        return

    document_paths = discover_documents()
    st.session_state.available_documents = [os.path.basename(path) for path in document_paths]

    if not document_paths:
        st.warning("⚠️ No documents found in the documents directory. Please add some documents to get started.")
        return

    # Check if documents are already processed
    try:
        collection = get_vector_collection()
        existing_docs = collection.get(limit=1000)
        existing_files = set()
        for metadata in existing_docs.get('metadatas', []):
            if metadata and 'source_file' in metadata:
                existing_files.add(metadata['source_file'])

        new_documents = [path for path in document_paths 
                        if os.path.basename(path) not in existing_files]

        if new_documents:
            progress_bar = st.progress(0)
            st.info(f"📚 Loading {len(new_documents)} new documents...")

            for i, doc_path in enumerate(new_documents):
                file_name = os.path.basename(doc_path)
                try:
                    chunks = process_backend_document(doc_path)
                    if chunks:
                        success = add_to_vector_collection(chunks, file_name)
                        if success:
                            st.success(f"✅ Loaded: {file_name}")
                    else:
                        st.warning(f"⚠️ No content extracted from: {file_name}")
                except Exception as e:
                    st.error(f"❌ Failed to load {file_name}: {e}")

                progress_bar.progress((i + 1) / len(new_documents))

            progress_bar.empty()
        else:
            st.info("📚 All documents are already loaded in the vector store.")
            # Update processed files from existing metadata
            for metadata in existing_docs.get('metadatas', []):
                if metadata and 'source_file' in metadata:
                    st.session_state.processed_files.add(metadata['source_file'])

    except Exception as e:
        st.error(f"Error checking existing documents: {e}")

    st.session_state.documents_loaded = True

def clear_vector_store():
    """Clear all data from vector store"""
    try:
        client = chromadb.PersistentClient(path="./demo-rag-chroma")
        client.delete_collection("rag_app")
        st.session_state.processed_files.clear()
        st.session_state.conversation_history.clear()
        st.session_state.documents_loaded = False
        st.success("🗑️ Vector store cleared successfully!")
        return True
    except Exception as e:
        st.error(f"Error clearing vector store: {e}")
        return False

# ============================== #
# Optimized Query and Ranking
# ============================== #
def query_collection(
    prompt: str, 
    n_results: int = 8,
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
    """Optimized re-ranking with proper confidence score normalization"""
    encoder = get_cross_encoder()
    
    # Limit document length for faster processing
    truncated_docs = [doc[:1000] for doc in documents]
    pairs = [(prompt, doc) for doc in truncated_docs]
    raw_scores = encoder.predict(pairs)
    
    # Convert raw scores to proper confidence scores
    confidence_scores = [float(1 / (1 + np.exp(-score))) for score in raw_scores]
    
    # Get top results based on raw scores
    top_results = sorted(enumerate(raw_scores), key=lambda x: x[1], reverse=True)[:top_k]
    
    top_ids = [i for i, _ in top_results]
    top_confidence_scores = [confidence_scores[i] for i in top_ids]
    top_text = "\n\n".join([documents[i] for i in top_ids])
    
    return top_text, top_ids, top_confidence_scores

# ============================== #
# Conversation Management
# ============================== #
def add_to_conversation(question: str, answer: str, confidence: float):
    """Add to conversation with size limit"""
    conversation_entry = {
        'timestamp': datetime.now().isoformat(),
        'question': question,
        'answer': answer[:500] + '...' if len(answer) > 500 else answer,
        'confidence': confidence
    }
    st.session_state.conversation_history.append(conversation_entry)
    
    # Keep only last 5 conversations
    if len(st.session_state.conversation_history) > 5:
        st.session_state.conversation_history = st.session_state.conversation_history[-5:]

def get_conversation_context() -> str:
    """Get minimal conversation context"""
    if not st.session_state.conversation_history:
        return ""
    
    context_parts = []
    for entry in st.session_state.conversation_history[-2:]:
        question = entry['question'][:100]
        answer = entry['answer'][:200]
        context_parts.append(f"Q: {question}\nA: {answer}")
    
    return "\n\n".join(context_parts)

# ============================== #
# LLM Call
# ============================== #
def call_llm(context: str, prompt: str, conversation_context: str = ""):
    """Optimized LLM call with shorter context"""
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
            "num_predict": 500,
        }
    )
    for chunk in response:
        if not chunk.get("done"):
            yield chunk["message"]["content"]

# ============================== #
# FAQ Display Function
# ============================== #
def display_faq():
    """Display FAQ section"""
    st.title("❓ Frequently Asked Questions")
    st.markdown("*School Safety Management - Common Questions and Answers*")
    
    # Search functionality for FAQ
    search_term = st.text_input("🔍 Search FAQ:", placeholder="Enter keywords to search...")
    
    # Filter FAQs based on search
    if search_term:
        filtered_faqs = [
            faq for faq in FAQ_DATA 
            if search_term.lower() in faq["question"].lower() or 
               search_term.lower() in faq["answer"].lower()
        ]
        st.info(f"Found {len(filtered_faqs)} results for '{search_term}'")
    else:
        filtered_faqs = FAQ_DATA
    
    # Display FAQs
    for i, faq in enumerate(filtered_faqs, 1):
        with st.expander(f"**{i}. {faq['question']}**"):
            st.write(faq["answer"])
    
    if search_term and not filtered_faqs:
        st.warning("No FAQ found matching your search. Try different keywords.")

# ============================== #
# AI Assistant Function
# ============================== #
def display_ai_assistant():
    """Display AI Assistant with complete functionality"""
    # Sidebar for system status and document management
    st.sidebar.header("📊 System Status")

    # Ollama status check
    try:
        collection = get_vector_collection()
        st.sidebar.success("🤖 Models Ready")
    except Exception as e:
        st.sidebar.error("❌ Model Issue")
        st.sidebar.write(f"Error: {str(e)[:50]}...")
        st.sidebar.markdown("**Quick fix:**")
        st.sidebar.code("ollama pull nomic-embed-text", language="bash")

    # Document loading section
    st.sidebar.subheader("📁 Document Library")

    # Load backend documents automatically
    with st.sidebar:
        if st.button("🔄 Refresh Documents", type="secondary"):
            st.session_state.documents_loaded = False
            st.session_state.processed_files.clear()

        load_backend_documents()

    # Show document statistics
    if st.session_state.available_documents:
        st.sidebar.success(f"📚 {len(st.session_state.available_documents)} documents available")
        with st.sidebar.expander("📋 Document List"):
            for doc in st.session_state.available_documents:
                status = "✅" if doc in st.session_state.processed_files else "⏳"
                st.write(f"{status} {doc}")
    else:
        st.sidebar.info("📂 No documents found")
        st.sidebar.markdown(f"**Add documents to:** `{DOCUMENTS_DIR}/`")

    # Clear vector store option
    if st.sidebar.button("🗑️ Clear Vector Store", type="secondary"):
        clear_vector_store()

    # Main app content
    st.markdown("*Ask questions about fire safety protocols, disaster management guidelines, and training programs*")

    # Document filters
    available_files, available_types = get_available_files_and_types()
    col1, col2 = st.columns(2)
    with col1:
        file_filter = st.selectbox(
            "Filter by document:",
            ["All Documents"] + available_files,
            index=0
        )
    with col2:
        file_type_filter = st.selectbox(
            "Filter by type:",
            ["All Types"] + available_types,
            index=0
        )

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about school safety policies..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Query documents
            results = query_collection(
                prompt,
                file_filter=file_filter if file_filter != "All Documents" else None,
                file_type_filter=file_type_filter if file_type_filter != "All Types" else None
            )

            # Rerank results
            context, _, confidence_scores = re_rank_cross_encoders(
                prompt,
                results['documents'][0] if results['documents'] else [""],
                top_k=3
            )

            # Get conversation context
            conversation_context = get_conversation_context()

            # Generate response
            for response in call_llm(context, prompt, conversation_context):
                full_response += response
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            st.caption(f"Confidence: {avg_confidence:.0%}")

            # Add to conversation history
            add_to_conversation(prompt, full_response, avg_confidence)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

# ============================== #
# Main App
# ============================== #
def main():
    # Page config
    st.set_page_config(
        page_title="School Safety Management",
        page_icon="🏫",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    # Create navbar
    create_navbar()

    # Display content based on active tab
    if st.session_state.active_tab == "AI Assistant":
        display_ai_assistant()
    else:
        display_faq()

if __name__ == "__main__":
    main()
