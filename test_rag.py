"""
Test suite for the Enterprise RAG Knowledge Base.
"""
import pytest
from fastapi.testclient import TestClient

from main import app
from vector_store import QdrantVectorStore
from rag_chain import RAGChain
from evaluator import RAGEvaluator


client = TestClient(app)


class TestHealthCheck:
    """Test health check endpoint."""
    
    def test_health_check(self):
        """Test that health check returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()


class TestIngest:
    """Test document ingestion."""
    
    def test_ingest_documents(self):
        """Test ingesting documents."""
        response = client.post(
            "/ingest",
            json={
                "texts": [
                    "Test document 1",
                    "Test document 2"
                ],
                "metadatas": [
                    {"source": "test1.txt"},
                    {"source": "test2.txt"}
                ]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "document_ids" in data
        assert "count" in data
        assert data["count"] > 0


class TestQuery:
    """Test query endpoints."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test data."""
        # Ingest test documents
        client.post(
            "/ingest",
            json={
                "texts": [
                    "Python is a programming language.",
                    "JavaScript is used for web development."
                ]
            }
        )
    
    def test_query(self):
        """Test basic query."""
        response = client.post(
            "/query",
            json={
                "question": "What is Python?",
                "return_contexts": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "contexts" in data
    
    def test_batch_query(self):
        """Test batch query."""
        response = client.post(
            "/query/batch",
            json=[
                "What is Python?",
                "What is JavaScript?"
            ]
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2


class TestVectorStore:
    """Test vector store functionality."""
    
    def test_add_documents(self):
        """Test adding documents to vector store."""
        vector_store = QdrantVectorStore()
        
        doc_ids = vector_store.add_documents(
            documents=["Test document"],
            metadatas=[{"source": "test.txt"}]
        )
        
        assert len(doc_ids) > 0
    
    def test_hybrid_search(self):
        """Test hybrid search."""
        vector_store = QdrantVectorStore()
        
        # Add documents
        vector_store.add_documents(
            documents=[
                "Machine learning is a subset of AI",
                "Deep learning uses neural networks"
            ]
        )
        
        # Search
        results = vector_store.hybrid_search(
            query="What is machine learning?",
            top_k=2
        )
        
        assert len(results) > 0
        assert "score" in results[0]
        assert "text" in results[0]


class TestRAGChain:
    """Test RAG chain functionality."""
    
    def test_query(self):
        """Test RAG chain query."""
        vector_store = QdrantVectorStore()
        rag_chain = RAGChain(vector_store)
        
        # Add test data
        vector_store.add_documents(
            documents=["AI is artificial intelligence"]
        )
        
        # Query
        result = rag_chain.query(
            question="What is AI?",
            return_contexts=True
        )
        
        assert "answer" in result
        assert "contexts" in result


class TestEvaluator:
    """Test evaluation functionality."""
    
    def test_evaluate(self):
        """Test Ragas evaluation."""
        evaluator = RAGEvaluator()
        
        scores = evaluator.evaluate(
            questions=["What is Python?"],
            answers=["Python is a programming language"],
            contexts=[["Python is a high-level programming language"]]
        )
        
        assert isinstance(scores, dict)
        assert len(scores) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
