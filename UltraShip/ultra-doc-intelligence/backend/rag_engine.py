import numpy as np
from typing import List, Tuple, Dict, Any
import openai  # Still using OpenAI SDK!
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

class RAGEngine:
    def __init__(self, document_processor):
        self.doc_processor = document_processor
        self.embedding_model = document_processor.embedding_model
        
        # Configure OpenAI client to use Groq's API endpoint
        self.client = openai.OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),  # Your Groq API key
            base_url="https://api.groq.com/openai/v1"  # Groq's OpenAI-compatible endpoint
        )
        
        # Model selection (choose based on your needs)
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "1024"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.1"))
        
        # Confidence thresholds
        self.high_confidence_threshold = 0.7
        self.medium_confidence_threshold = 0.5
        self.low_confidence_threshold = 0.3
    
    def retrieve_relevant_chunks(self, query: str, file_id: str, top_k: int = 3) -> List[Tuple[str, float, int]]:
        """Retrieve most relevant chunks for a query"""
        
        # Load document
        doc = self.doc_processor.load_document(file_id)
        if not doc:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Search in FAISS index
        index = doc['index']
        distances, indices = index.search(query_embedding.astype('float32'), top_k)
        
        # Convert distances to similarity scores
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(doc['chunks']):
                # Convert distance to similarity (inverse relationship)
                # Groq embeddings are cosine-based, so we use this conversion
                similarity = 1 / (1 + distance)  # Simple conversion
                
                results.append((
                    doc['chunks'][idx],
                    float(similarity),
                    int(idx)
                ))
        
        return results
    
    def generate_answer(self, query: str, context_chunks: List[Tuple[str, float, int]]) -> Dict[str, Any]:
        """Generate answer using Groq"""
        
        if not context_chunks:
            return {
                'answer': "No relevant information found in the document.",
                'confidence': 0.0,
                'source_text': "",
                'source_chunk_index': -1
            }
        
        # Prepare context
        context = "\n\n".join([chunk for chunk, _, _ in context_chunks])
        best_chunk, best_score, best_idx = context_chunks[0]
        
        # Guardrail 1: Check if similarity is too low
        if best_score < self.low_confidence_threshold:
            return {
                'answer': "I cannot confidently answer this question based on the document content.",
                'confidence': best_score,
                'source_text': best_chunk[:500] + "..." if len(best_chunk) > 500 else best_chunk,
                'source_chunk_index': best_idx
            }
        
        try:
            # System prompt for better extraction
            system_prompt = """You are a logistics document expert. Your task is to answer questions based ONLY on the provided context.
            
            RULES:
            1. Only use information from the context
            2. If the answer isn't in the context, say "Not found in document"
            3. Be concise and specific
            4. Include relevant details like dates, names, and numbers when available
            5. Never make up information"""
            
            # User prompt with context and question
            user_prompt = f"""CONTEXT:
{context}

QUESTION: {query}

ANSWER (based only on the context above):"""
            
            # Make API call to Groq
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=1,
                stream=False
            )
            
            answer = response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Groq API error: {e}")
            # Fallback to simple extraction
            answer = self.extract_answer_from_context(query, context)
        
        # Calculate confidence score
        confidence = self.calculate_confidence(answer, context_chunks, query)
        
        # Guardrail 2: Check if answer indicates not found
        if "not found" in answer.lower() or "cannot answer" in answer.lower():
            confidence = min(confidence, 0.3)
        
        return {
            'answer': answer,
            'confidence': confidence,
            'source_text': best_chunk[:500] + "..." if len(best_chunk) > 500 else best_chunk,
            'source_chunk_index': best_idx
        }
    
    def extract_answer_from_context(self, query: str, context: str) -> str:
        """Simple keyword-based answer extraction (fallback)"""
        # Extract sentences containing query keywords
        sentences = context.split('. ')
        keywords = set(query.lower().split())
        
        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in keywords):
                relevant_sentences.append(sentence)
        
        if relevant_sentences:
            return ". ".join(relevant_sentences[:2]) + "."
        else:
            return "Not found in document."
    
    def calculate_confidence(self, answer: str, context_chunks: List[Tuple[str, float, int]], query: str) -> float:
        """Calculate confidence score based on multiple factors"""
        
        if not context_chunks:
            return 0.0
        
        # Factor 1: Retrieval similarity (weight: 0.5)
        similarity_score = context_chunks[0][1]
        
        # Factor 2: Answer coverage (weight: 0.3)
        # Check how much of the answer appears in the source chunk
        best_chunk = context_chunks[0][0]
        answer_words = set(answer.lower().split())
        chunk_words = set(best_chunk.lower().split())
        
        if answer_words:
            coverage = len(answer_words.intersection(chunk_words)) / len(answer_words)
        else:
            coverage = 0
        
        # Factor 3: Multiple chunk agreement (weight: 0.2)
        if len(context_chunks) > 1:
            chunk1_words = set(context_chunks[0][0].lower().split())
            chunk2_words = set(context_chunks[1][0].lower().split())
            if chunk1_words and chunk2_words:
                agreement = len(chunk1_words.intersection(chunk2_words)) / len(chunk1_words.union(chunk2_words))
            else:
                agreement = 0.5
        else:
            agreement = 0.5  # Neutral if only one chunk
        
        # Weighted combination
        confidence = (0.5 * similarity_score) + (0.3 * coverage) + (0.2 * agreement)
        
        # Guardrail: If answer indicates missing info, reduce confidence
        if "not found" in answer.lower() or "cannot answer" in answer.lower():
            confidence *= 0.3
        
        return min(max(confidence, 0.0), 1.0)  # Clamp to [0, 1]