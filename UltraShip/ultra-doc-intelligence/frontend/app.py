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
    # For Streamlit Cloud, we'll use a relative URL or you can set this in secrets
    st.session_state.backend_url = st.secrets.get("BACKEND_URL", "http://backend:8000")

# Title
st.title("📄 Ultra Doc-Intelligence")
st.markdown("Upload logistics documents and ask questions about them")

# Sidebar
with st.sidebar:
    st.header("Document Upload")
    
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
                    # Include API key in headers (or use environment variables on backend)
                    headers = {
                        "X-API-Key": st.secrets.get("GROQ_API_KEY", "")
                    }
                    
                    response = requests.post(
                        f"{st.session_state.backend_url}/upload", 
                        files=files,
                        headers=headers,
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
                except Exception as e:
                    st.error(f"Connection error: {e}")
    
    # Display current document
    if st.session_state.file_id:
        st.info(f"📄 Current document: **{st.session_state.filename}**")
        
        # Structured extraction button
        if st.button("🔍 Extract Structured Data", type="secondary"):
            with st.spinner("Extracting data..."):
                try:
                    headers = {
                        "X-API-Key": st.secrets.get("GROQ_API_KEY", "")
                    }
                    
                    response = requests.post(
                        f"{st.session_state.backend_url}/extract",
                        params={"file_id": st.session_state.file_id},
                        headers=headers,
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
                headers = {
                    "X-API-Key": st.secrets.get("GROQ_API_KEY", "")
                }
                
                response = requests.post(
                    f"{st.session_state.backend_url}/ask",
                    json={
                        "file_id": st.session_state.file_id,
                        "question": question
                    },
                    headers=headers,
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
        st.json(data)
        
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

# import streamlit as st
# import requests
# import json
# from datetime import datetime

# # Configuration
# API_BASE_URL = "http://127.0.0.1:8000"  # Change this to your backend URL

# st.set_page_config(
#     page_title="Ultra Doc-Intelligence",
#     page_icon="📄",
#     layout="wide"
# )

# # Initialize session state
# if 'file_id' not in st.session_state:
#     st.session_state.file_id = None
# if 'filename' not in st.session_state:
#     st.session_state.filename = None
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []

# # Title
# st.title("📄 Ultra Doc-Intelligence")
# st.markdown("Upload logistics documents and ask questions about them")

# # Sidebar
# with st.sidebar:
#     st.header("Document Upload")
    
#     uploaded_file = st.file_uploader(
#         "Choose a file",
#         type=['pdf', 'docx', 'txt'],
#         help="Upload PDF, DOCX, or TXT files"
#     )
    
#     if uploaded_file is not None:
#         if st.button("Process Document", type="primary"):
#             with st.spinner("Processing document..."):
#                 # Upload to backend
#                 files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
#                 try:
#                     response = requests.post(f"{API_BASE_URL}/upload", files=files)
#                     if response.status_code == 200:
#                         data = response.json()
#                         st.session_state.file_id = data['file_id']
#                         st.session_state.filename = data['filename']
#                         st.success(f"✅ Document processed: {data['chunks']} chunks created")
#                     else:
#                         st.error(f"Error: {response.text}")
#                 except Exception as e:
#                     st.error(f"Connection error: {e}")
    
#     # Display current document
#     if st.session_state.file_id:
#         st.info(f"📄 Current document: **{st.session_state.filename}**")
        
#         # Structured extraction button
#         if st.button("🔍 Extract Structured Data", type="secondary"):
#             with st.spinner("Extracting data..."):
#                 try:
#                     response = requests.post(
#                         f"{API_BASE_URL}/extract",
#                         params={"file_id": st.session_state.file_id}
#                     )
#                     if response.status_code == 200:
#                         data = response.json()
#                         st.session_state.extracted_data = data
#                         st.success("Extraction complete!")
#                     else:
#                         st.error(f"Error: {response.text}")
#                 except Exception as e:
#                     st.error(f"Connection error: {e}")

# # Main content area
# col1, col2 = st.columns([2, 1])

# with col1:
#     st.header("💬 Ask Questions")
    
#     # Question input
#     question = st.text_input(
#         "Ask about your document",
#         placeholder="e.g., What is the pickup date? Who is the carrier?",
#         disabled=not st.session_state.file_id
#     )
    
#     if question and st.session_state.file_id:
#         with st.spinner("Finding answer..."):
#             try:
#                 response = requests.post(
#                     f"{API_BASE_URL}/ask",
#                     json={
#                         "file_id": st.session_state.file_id,
#                         "question": question
#                     }
#                 )
                
#                 if response.status_code == 200:
#                     result = response.json()
                    
#                     # Add to chat history
#                     st.session_state.chat_history.append({
#                         "question": question,
#                         "answer": result['answer'],
#                         "confidence": result['confidence'],
#                         "source": result['source_text'],
#                         "timestamp": datetime.now().strftime("%H:%M:%S")
#                     })
#                 else:
#                     st.error(f"Error: {response.text}")
                    
#             except Exception as e:
#                 st.error(f"Connection error: {e}")
    
#     # Display chat history
#     for chat in reversed(st.session_state.chat_history):
#         with st.container():
#             # Question
#             st.markdown(f"**Q: {chat['question']}**")
            
#             # Answer with confidence indicator
#             confidence = chat['confidence']
#             if confidence >= 0.7:
#                 conf_color = "green"
#                 conf_label = "High"
#             elif confidence >= 0.4:
#                 conf_color = "orange"
#                 conf_label = "Medium"
#             else:
#                 conf_color = "red"
#                 conf_label = "Low"
            
#             st.markdown(
#                 f"**A:** {chat['answer']}  \n"
#                 f"<span style='color:{conf_color}'>● Confidence: {conf_label} ({confidence:.2f})</span>",
#                 unsafe_allow_html=True
#             )
            
#             # Expandable source
#             with st.expander("View source text"):
#                 st.text(chat['source'])
            
#             st.divider()

# with col2:
#     st.header("📋 Extracted Data")
    
#     if 'extracted_data' in st.session_state and st.session_state.extracted_data:
#         data = st.session_state.extracted_data
        
#         # Display in a nice format
#         st.json(data)
        
#         # Download button
#         if st.button("📥 Download JSON"):
#             json_str = json.dumps(data, indent=2)
#             st.download_button(
#                 label="Download",
#                 data=json_str,
#                 file_name=f"extracted_{st.session_state.file_id}.json",
#                 mime="application/json"
#             )
#     else:
#         st.info("Click 'Extract Structured Data' in the sidebar to extract shipment information")

# # Footer
# st.markdown("---")
# st.markdown("Ultra Doc-Intelligence - AI-powered logistics document assistant")