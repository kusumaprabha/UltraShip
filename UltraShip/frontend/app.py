import streamlit as st
import requests
import json
from datetime import datetime
import os

# Configuration - will be set by Streamlit Cloud secrets
st.set_page_config(
    page_title="Ultra Doc-Intelligence",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'file_id' not in st.session_state:
    st.session_state.file_id = None
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None
if 'backend_url' not in st.session_state:
    # First check Streamlit secrets, then fallback to environment variable, then default to Render URL
    st.session_state.backend_url = (
        st.secrets.get("BACKEND_URL") or 
        os.getenv("BACKEND_URL") or 
        "https://ultraship-s1xq.onrender.com"
    )

# Title
st.title("📄 Ultra Doc-Intelligence")
st.markdown("Upload logistics documents and ask questions about them")

# Sidebar
with st.sidebar:
    st.header("Document Upload")
    
    # Show backend connection status
    try:
        health_check = requests.get(f"{st.session_state.backend_url}/health", timeout=5)
        if health_check.status_code == 200:
            st.success("✅ Connected to backend")
        else:
            st.warning("⚠️ Backend connection issue")
    except:
        st.error("❌ Cannot connect to backend")
        st.info(f"Trying to connect to: {st.session_state.backend_url}")
    
    # Show that API key is configured (without revealing it)
    if st.secrets.get("GROQ_API_KEY"):
        st.success("✅ Groq API configured")
    else:
        st.error("❌ Groq API key not found in secrets")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'docx', 'txt'],
        help="Upload PDF, DOCX, or TXT files"
    )
    
    if uploaded_file is not None:
        if st.button("Process Document", type="primary"):
            with st.spinner("Processing document..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                try:
                    # Note: API key is not sent to backend - backend reads it from its own environment
                    response = requests.post(
                        f"{st.session_state.backend_url}/upload", 
                        files=files,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.file_id = data['file_id']
                        st.session_state.filename = data['filename']
                        st.success(f"✅ Document processed: {data['chunks']} chunks created")
                        st.rerun()
                    else:
                        st.error(f"Error: {response.text}")
                except requests.exceptions.ConnectionError as e:
                    st.error(f"Connection error: Cannot reach backend at {st.session_state.backend_url}")
                    st.error("Make sure your backend is running and the URL is correct")
                except requests.exceptions.Timeout:
                    st.error("Connection timeout: Backend took too long to respond")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
    
    # Display current document
    if st.session_state.file_id:
        st.info(f"📄 Current document: **{st.session_state.filename}**")
        
        # Structured extraction button
        if st.button("🔍 Extract Structured Data", type="secondary"):
            with st.spinner("Extracting data..."):
                try:
                    response = requests.post(
                        f"{st.session_state.backend_url}/extract",
                        params={"file_id": st.session_state.file_id},
                        timeout=30
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.extracted_data = data
                        st.success("Extraction complete!")
                        st.rerun()
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Ask Questions")
    
    question = st.text_input(
        "Ask about your document",
        placeholder="e.g., What is the pickup date? Who is the carrier?",
        disabled=not st.session_state.file_id
    )
    
    if question and st.session_state.file_id:
        with st.spinner("Finding answer..."):
            try:
                response = requests.post(
                    f"{st.session_state.backend_url}/ask",
                    json={
                        "file_id": st.session_state.file_id,
                        "question": question
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": result['answer'],
                        "confidence": result['confidence'],
                        "source": result['source_text'],
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    st.rerun()
                else:
                    st.error(f"Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Connection error: {e}")
    
    # Display chat history
    for chat in reversed(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**Q: {chat['question']}**")
            
            confidence = chat['confidence']
            if confidence >= 0.7:
                conf_color = "green"
                conf_label = "High"
            elif confidence >= 0.4:
                conf_color = "orange"
                conf_label = "Medium"
            else:
                conf_color = "red"
                conf_label = "Low"
            
            st.markdown(
                f"**A:** {chat['answer']}  \n"
                f"<span style='color:{conf_color}'>● Confidence: {conf_label} ({confidence:.2f})</span>",
                unsafe_allow_html=True
            )
            
            with st.expander("View source text"):
                st.text(chat['source'])
            
            st.divider()

with col2:
    st.header("📋 Extracted Data")
    
    if 'extracted_data' in st.session_state and st.session_state.extracted_data:
        data = st.session_state.extracted_data
        
        # Display as formatted JSON
        st.json(data)
        
        # Download button
        if st.button("📥 Download JSON"):
            json_str = json.dumps(data, indent=2)
            st.download_button(
                label="Download",
                data=json_str,
                file_name=f"extracted_{st.session_state.file_id}.json",
                mime="application/json"
            )
    else:
        st.info("Click 'Extract Structured Data' in the sidebar")

# Footer
st.markdown("---")
st.markdown("Ultra Doc-Intelligence - AI-powered logistics document assistant")

# Debug information (optional - remove in production)
with st.expander("Debug Info", expanded=False):
    st.write("Backend URL:", st.session_state.backend_url)
    st.write("Session State:", {
        "file_id": st.session_state.file_id,
        "filename": st.session_state.filename,
        "chat_history_count": len(st.session_state.chat_history),
        "has_extracted_data": st.session_state.extracted_data is not None
    })
