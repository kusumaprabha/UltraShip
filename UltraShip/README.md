# Ultra Doc-Intelligence

An AI-powered document intelligence system for logistics documents. Upload PDFs, DOCXs, or TXT files and ask questions about their content with grounded answers and confidence scoring.

## Features

- **Document Processing**: Upload and process logistics documents (Rate Confirmations, BOLs, Invoices)
- **Intelligent Q&A**: Ask natural language questions about document content
- **Confidence Scoring**: Each answer includes a confidence score (0-1)
- **Guardrails**: Prevents hallucinations by refusing low-confidence answers
- **Structured Extraction**: Extract shipment data into JSON format
- **Minimal UI**: Simple Streamlit interface for testing

## 1.Architecture

[Streamlit UI] ↔ [FastAPI Backend] ↔ [Document Processor]
├── Text Extraction
├── Intelligent Chunking
├── Embeddings (all-MiniLM-L6-v2)
├── Vector Search (FAISS)
└── LLM (OpenAI/Simple Extraction)


##  Installation

### Prerequisites
- Python 3.9+
- pip


# Install backend dependencies

pip install -r requirements.txt

# Create .env file with Groq API key (optional)
echo "GROQ_API_KEY=your_key_here" > .env

# Run backend
python app.py

# Frontend Setup
# In a new terminal

pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py


## Usage
Open frontend at http://127.0.0.1:8000

Upload a logistics document (PDF, DOCX, TXT)

Wait for processing

Ask questions about the document

View answers with confidence scores and sources

Extract structured shipment data


### Key Implementation Details

## 2.Chunking Strategy 

Size: 500 words per chunk with 100-word overlap

Intelligent splitting: Attempts to break at sentence boundaries

Overlap: Ensures context continuity between chunks

## 3.Retrieval Method

Embeddings: SentenceTransformer 'all-MiniLM-L6-v2' (384-dim)

Vector Store: FAISS for fast similarity search

Top-k: Retrieves top 3 most relevant chunks

## 4.Guardrails Approach

Similarity Threshold: Refuse if best chunk similarity < 0.3

Answer Verification: Check if answer contains "not found" phrases

Confidence Scoring: Multi-factor confidence calculation

## 5.Confidence Scoring Formula
text
Confidence = (0.5 × Similarity) + (0.3 × Coverage) + (0.2 × Agreement)
Similarity: Query-chunk embedding similarity

Coverage: Answer overlap with source chunk

Agreement: Consistency across multiple retrieved chunks

Example Questions

"What is the pickup date?"

"Who is the carrier?"

"How much does it weigh?"

"What is the rate?"

"Where is it going?"

### 6. Failure Cases & Limitations

Poor Quality PDFs: Scanned documents without OCR won't extract text well

Ambiguous Questions: Vague questions may get low-confidence answers

Missing Information: System correctly returns "Not found" when data missing

Complex Tables: Table extraction is basic; complex tables may be misinterpreted

OpenAI Dependency: Without API key, falls back to simple keyword extraction

### 7. Improvement Ideas 

Layout-Aware Chunking: Use document layout (tables, headers) for better chunking

Fine-tuned Extraction Model: Train a small model for field extraction

Multi-Document Support: Query across multiple documents

Better OCR: Integrate Tesseract for scanned documents

Caching: Cache embeddings for faster repeated queries

Streaming Responses: For longer answers

More Guardrails: Add fact-checking against document

🧪 Testing
Test with sample documents from:
AI Engineer Test - Sample Documents

📝 License
MIT


## Running the Application**

### **Terminal 1 - Backend**

python app.py

Server runs on http://127.0.0.1:8000

Terminal 2 - Frontend

streamlit run app.py

UI runs on http://127.0.0.1:8000

###  Testing the System

Upload a document from the sample folder

Ask questions like:

"What is the carrier rate?"

"When is pickup scheduled?"

"Who is the consignee?"

Check confidence scores - they should be higher for well-matched answers

Test guardrails - ask about something not in the document

Extract structured data - verify the JSON output

This complete solution provides:

✅ Full backend API with FastAPI

✅ Document processing with intelligent chunking

✅ RAG-based Q&A with confidence scoring

✅ Multiple guardrails

✅ Structured extraction

✅ Streamlit UI

✅ Comprehensive README

✅ Local-first approach (OpenAI optional)

The system is modular, well-documented, and ready for deployment. 




