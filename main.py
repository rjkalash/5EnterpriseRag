"""
FastAPI application with LangServe integration for Enterprise RAG Knowledge Base.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import tempfile
from pathlib import Path
from loguru import logger

from langserve import add_routes

from config import settings
from vector_store import QdrantVectorStore
from rag_chain import RAGChain
from evaluator import RAGEvaluator
from document_processor import DocumentProcessor


# Initialize components
vector_store = QdrantVectorStore()
rag_chain = RAGChain(vector_store)
evaluator = RAGEvaluator()
doc_processor = DocumentProcessor()


# Pydantic models
class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    question: str = Field(..., description="Question to ask")
    top_k: Optional[int] = Field(None, description="Number of contexts to retrieve")
    return_contexts: bool = Field(False, description="Whether to return contexts")


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str
    contexts: Optional[List[Dict[str, Any]]] = None


class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    texts: List[str] = Field(..., description="List of document texts")
    metadatas: Optional[List[Dict[str, Any]]] = Field(None, description="Document metadata")


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    document_ids: List[str]
    count: int


class EvaluationRequest(BaseModel):
    """Request model for RAG evaluation."""
    questions: List[str]
    ground_truths: Optional[List[str]] = None


class EvaluationResponse(BaseModel):
    """Response model for evaluation."""
    scores: Dict[str, float]
    improvement_percentage: Optional[float] = None


# Create FastAPI app
app = FastAPI(
    title="Enterprise RAG Knowledge Base",
    description="Production-ready RAG system with hybrid search and evaluation",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        collection_info = vector_store.get_collection_info()
        return {
            "status": "healthy",
            "collection": collection_info,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system.
    
    Args:
        request: Query request with question and parameters
        
    Returns:
        Answer and optionally retrieved contexts
    """
    try:
        logger.info(f"Received query: {request.question[:50]}...")
        
        result = rag_chain.query(
            question=request.question,
            top_k=request.top_k,
            return_contexts=request.return_contexts,
        )
        
        return QueryResponse(**result)
    
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Batch query endpoint
@app.post("/query/batch")
async def batch_query(questions: List[str], top_k: Optional[int] = None):
    """
    Process multiple queries in batch.
    
    Args:
        questions: List of questions
        top_k: Number of contexts per question
        
    Returns:
        List of answers with contexts
    """
    try:
        logger.info(f"Received batch query with {len(questions)} questions")
        
        results = rag_chain.batch_query(questions=questions, top_k=top_k)
        
        return {"results": results, "count": len(results)}
    
    except Exception as e:
        logger.error(f"Batch query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Ingest endpoint
@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """
    Ingest documents into the knowledge base.
    
    Args:
        request: Ingest request with texts and metadata
        
    Returns:
        Document IDs and count
    """
    try:
        logger.info(f"Ingesting {len(request.texts)} documents")
        
        doc_ids = vector_store.add_documents(
            documents=request.texts,
            metadatas=request.metadatas,
        )
        
        return IngestResponse(
            document_ids=doc_ids,
            count=len(doc_ids),
        )
    
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# File upload endpoint
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and process a document file.
    
    Args:
        file: Uploaded file
        
    Returns:
        Processing result with document IDs
    """
    try:
        logger.info(f"Received file upload: {file.filename}")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Process file
            documents = doc_processor.load_file(tmp_path)
            texts, metadatas = doc_processor.extract_text_and_metadata(documents)
            
            # Ingest into vector store
            doc_ids = vector_store.add_documents(
                documents=texts,
                metadatas=metadatas,
            )
            
            return {
                "filename": file.filename,
                "document_count": len(documents),
                "chunk_count": len(doc_ids),
                "document_ids": doc_ids,
            }
        
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
    
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Evaluation endpoint
@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_system(request: EvaluationRequest):
    """
    Evaluate RAG system performance.
    
    Args:
        request: Evaluation request with questions and ground truths
        
    Returns:
        Evaluation scores
    """
    try:
        if not settings.enable_evaluation:
            raise HTTPException(
                status_code=403,
                detail="Evaluation is disabled in settings"
            )
        
        logger.info(f"Evaluating system with {len(request.questions)} questions")
        
        # Generate answers
        results = rag_chain.batch_query(request.questions)
        
        answers = [r["answer"] for r in results]
        contexts = [[ctx["text"] for ctx in r["contexts"]] for r in results]
        
        # Evaluate
        scores = evaluator.evaluate(
            questions=request.questions,
            answers=answers,
            contexts=contexts,
            ground_truths=request.ground_truths,
        )
        
        return EvaluationResponse(scores=scores)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Collection info endpoint
@app.get("/collection/info")
async def get_collection_info():
    """Get information about the vector collection."""
    try:
        info = vector_store.get_collection_info()
        return info
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Collection reset endpoint
@app.delete("/collection/reset")
async def reset_collection():
    """Delete and recreate the collection (use with caution!)."""
    try:
        logger.warning("Resetting collection")
        vector_store.delete_collection()
        vector_store._ensure_collection()
        return {"status": "Collection reset successfully"}
    except Exception as e:
        logger.error(f"Failed to reset collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Add LangServe routes for the RAG chain
add_routes(
    app,
    rag_chain.chain,
    path="/langserve",
    playground_type="default",
)


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("=" * 50)
    logger.info("Enterprise RAG Knowledge Base Starting")
    logger.info("=" * 50)
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Qdrant Host: {settings.qdrant_host}:{settings.qdrant_port}")
    logger.info(f"Collection: {settings.qdrant_collection_name}")
    logger.info(f"LLM Model: {settings.openai_model}")
    logger.info(f"Embedding Model: {settings.embedding_model}")
    logger.info("=" * 50)


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Enterprise RAG Knowledge Base")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.environment == "development",
    )
