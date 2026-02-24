from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn
import logging
import traceback
import sys
import os
from models import *
from document_processor import DocumentProcessor
from rag_engine import RAGEngine
from extractor import StructuredExtractor

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('api_debug.log')
    ]
)
logger = logging.getLogger(__name__)

# Check for required environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY environment variable not set")
    # In production, you might want to raise an exception here
    # For now, we'll log an error but continue (the components will fail if they need it)
else:
    logger.info("GROQ_API_KEY found in environment variables")

# Initialize components with error handling
try:
    logger.info("Initializing DocumentProcessor...")
    doc_processor = DocumentProcessor()
    logger.info("DocumentProcessor initialized")
    
    logger.info("Initializing RAGEngine...")
    # Pass the API key to RAGEngine if it accepts it
    # You may need to modify your RAGEngine class to accept an API key parameter
    rag_engine = RAGEngine(doc_processor, groq_api_key=GROQ_API_KEY)
    logger.info("RAGEngine initialized")
    
    logger.info("Initializing StructuredExtractor...")
    # Pass the API key to StructuredExtractor if it accepts it
    extractor = StructuredExtractor(doc_processor, groq_api_key=GROQ_API_KEY)
    logger.info("StructuredExtractor initialized")
    
except Exception as e:
    logger.error(f"Failed to initialize components: {str(e)}")
    logger.error(traceback.format_exc())
    raise

@app.get("/")
async def root():
    return {"message": "Ultra Doc-Intelligence API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint to verify API is working"""
    return {
        "status": "healthy",
        "components": {
            "document_processor": "initialized",
            "rag_engine": "initialized",
            "extractor": "initialized"
        },
        "env_vars": {
            "groq_api_key_set": bool(GROQ_API_KEY)
        }
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a logistics document
    """
    request_id = f"upload_{id(file)}"
    logger.info(f"[{request_id}] Uploading file: {file.filename}")
    
    try:
        # Validate file
        if not file.filename:
            logger.error(f"[{request_id}] No filename provided")
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file size (limit to 10MB)
        content = await file.read()
        file_size = len(content)
        logger.info(f"[{request_id}] File size: {file_size} bytes")
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            logger.error(f"[{request_id}] File too large: {file_size} bytes")
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        # Reset file position for potential future reads
        await file.seek(0)
        
        # Process document
        logger.info(f"[{request_id}] Processing document...")
        result = doc_processor.process_document(content, file.filename)
        
        logger.info(f"[{request_id}] Document processed: {result['file_id']} with {result['chunks']} chunks")
        return UploadResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Upload error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question about an uploaded document
    """
    request_id = f"ask_{id(request)}"
    logger.info(f"[{request_id}] Question received: {request.question}")
    logger.info(f"[{request_id}] File ID: {request.file_id}")
    
    try:
        # Validate request
        if not request.file_id:
            logger.error(f"[{request_id}] No file_id provided")
            raise HTTPException(status_code=400, detail="file_id is required")
        
        if not request.question or not request.question.strip():
            logger.error(f"[{request_id}] No question provided")
            raise HTTPException(status_code=400, detail="question is required")
        
        # Check if document exists
        logger.info(f"[{request_id}] Loading document...")
        doc = doc_processor.load_document(request.file_id)
        
        if not doc:
            logger.error(f"[{request_id}] Document not found: {request.file_id}")
            raise HTTPException(
                status_code=404, 
                detail=f"Document with file_id '{request.file_id}' not found. Please upload the document first."
            )
        
        # Log document details
        chunks_count = len(doc.get('chunks', []))
        logger.info(f"[{request_id}] Document loaded: {chunks_count} chunks available")
        
        if chunks_count == 0:
            logger.warning(f"[{request_id}] Document has no chunks")
            return AskResponse(
                answer="The document appears to be empty or couldn't be processed.",
                confidence=0.0,
                source_text="",
                source_chunk_index=-1
            )
        
        # Retrieve relevant chunks
        logger.info(f"[{request_id}] Retrieving relevant chunks for: '{request.question}'")
        try:
            chunks = rag_engine.retrieve_relevant_chunks(
                query=request.question,
                file_id=request.file_id
            )
            logger.info(f"[{request_id}] Retrieved {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"[{request_id}] Error in retrieve_relevant_chunks: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error retrieving chunks: {str(e)}")
        
        if not chunks:
            logger.warning(f"[{request_id}] No relevant chunks found for query")
            return AskResponse(
                answer="No relevant information found in the document for your question.",
                confidence=0.0,
                source_text="",
                source_chunk_index=-1
            )
        
        # Log chunk details for debugging
        for i, (chunk_text, score, idx) in enumerate(chunks):
            logger.debug(f"[{request_id}] Chunk {i}: score={score:.3f}, index={idx}")
            logger.debug(f"[{request_id}] Chunk preview: {chunk_text[:150]}...")
        
        # Generate answer
        logger.info(f"[{request_id}] Generating answer...")
        try:
            result = rag_engine.generate_answer(request.question, chunks)
        except Exception as e:
            logger.error(f"[{request_id}] Error in generate_answer: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return a fallback response instead of failing
            return AskResponse(
                answer=f"Error generating answer: {str(e)}",
                confidence=0.0,
                source_text=chunks[0][0][:500] + "..." if chunks[0][0] else "",
                source_chunk_index=chunks[0][2] if chunks else -1
            )
        
        # Validate result
        if not result or 'answer' not in result:
            logger.error(f"[{request_id}] Invalid result from generate_answer: {result}")
            raise HTTPException(status_code=500, detail="Invalid response from answer generator")
        
        logger.info(f"[{request_id}] Answer generated: '{result['answer'][:100]}...'")
        logger.info(f"[{request_id}] Confidence: {result.get('confidence', 0):.3f}")
        
        return AskResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}]  Unexpected error in ask_question: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Ask failed: {str(e)}")

@app.post("/extract", response_model=ExtractionResponse)
async def extract_data(file_id: str):
    """
    Extract structured shipment data from document
    """
    request_id = f"extract_{id(file_id)}"
    logger.info(f"[{request_id}] Extracting data from file: {file_id}")
    
    try:
        # Validate
        if not file_id:
            logger.error(f"[{request_id}] No file_id provided")
            raise HTTPException(status_code=400, detail="file_id is required")
        
        # Check if document exists
        doc = doc_processor.load_document(file_id)
        if not doc:
            logger.error(f"[{request_id}] Document not found: {file_id}")
            raise HTTPException(status_code=404, detail=f"Document with file_id '{file_id}' not found")
        
        # Extract fields
        logger.info(f"[{request_id}] Starting extraction...")
        result = extractor.extract_fields(file_id)
        
        # Count extracted fields
        extracted_count = sum(1 for v in result.values() if v is not None)
        logger.info(f"[{request_id}] Extraction complete: {extracted_count} fields found")
        
        return ExtractionResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Extract error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

# Debug endpoint to list available documents
@app.get("/documents")
async def list_documents():
    """List all uploaded documents (for debugging)"""
    try:
        documents = []
        import os
        if os.path.exists(doc_processor.upload_dir):
            files = os.listdir(doc_processor.upload_dir)
            documents = [f for f in files if not f.startswith('.')]
        
        return {
            "count": len(documents),
            "documents": documents,
            "processed_ids": list(doc_processor.documents.keys())
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return {"error": str(e)}

# MAIN:
if __name__ == "__main__":
    logger.info("Starting Ultra Doc-Intelligence API server...")
    logger.info(f"Listening on http://127.0.0.1:8000")
    
    # Log environment info (without exposing the actual key)
    if GROQ_API_KEY:
        logger.info(f"GROQ_API_KEY is set (length: {len(GROQ_API_KEY)} characters)")
    else:
        logger.warning("GROQ_API_KEY is not set - API functionality may be limited")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug")
