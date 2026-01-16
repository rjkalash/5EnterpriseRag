"""
Qdrant vector store manager with hybrid search capabilities.
"""
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    ScoredPoint,
    SearchRequest,
    Prefetch,
    Query,
)
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
import uuid

from config import settings


class QdrantVectorStore:
    """Manages Qdrant vector store with hybrid search (sparse + dense)."""
    
    def __init__(self):
        """Initialize Qdrant client and embeddings."""
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
        )
        self.collection_name = settings.qdrant_collection_name
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if self.collection_name not in collection_names:
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=1536,  # OpenAI embedding dimension
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": {}  # BM25-like sparse vectors
                },
            )
            logger.info(f"Collection {self.collection_name} created successfully")
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Add documents to the vector store with chunking.
        
        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
            
        Returns:
            List of document IDs
        """
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )
        
        all_chunks = []
        all_metadatas = []
        
        for idx, doc in enumerate(documents):
            chunks = text_splitter.split_text(doc)
            all_chunks.extend(chunks)
            
            # Add metadata for each chunk
            doc_metadata = metadatas[idx] if metadatas else {}
            for chunk_idx, _ in enumerate(chunks):
                chunk_metadata = {
                    **doc_metadata,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "source_doc_index": idx,
                }
                all_metadatas.append(chunk_metadata)
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks")
        dense_vectors = self.embeddings.embed_documents(all_chunks)
        
        # Generate sparse vectors (simple BM25-like representation)
        sparse_vectors = self._generate_sparse_vectors(all_chunks)
        
        # Create points
        points = []
        doc_ids = []
        
        for idx, (chunk, dense_vec, sparse_vec, metadata) in enumerate(
            zip(all_chunks, dense_vectors, sparse_vectors, all_metadatas)
        ):
            point_id = str(uuid.uuid4())
            doc_ids.append(point_id)
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vec,
                        "sparse": sparse_vec,
                    },
                    payload={
                        "text": chunk,
                        **metadata,
                    },
                )
            )
        
        # Upload to Qdrant
        logger.info(f"Uploading {len(points)} points to Qdrant")
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        
        logger.info(f"Successfully added {len(doc_ids)} document chunks")
        return doc_ids
    
    def _generate_sparse_vectors(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Generate sparse vectors for BM25-like search.
        
        Args:
            texts: List of text chunks
            
        Returns:
            List of sparse vector representations
        """
        sparse_vectors = []
        
        for text in texts:
            # Simple term frequency representation
            words = text.lower().split()
            word_freq = {}
            
            for word in words:
                if len(word) > 2:  # Filter short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Convert to sparse vector format
            indices = []
            values = []
            
            for word, freq in sorted(word_freq.items()):
                # Use hash of word as index
                indices.append(hash(word) % 100000)
                values.append(float(freq))
            
            sparse_vectors.append({
                "indices": indices,
                "values": values,
            })
        
        return sparse_vectors
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining dense and sparse retrieval.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_conditions: Optional metadata filters
            
        Returns:
            List of search results with scores
        """
        if top_k is None:
            top_k = settings.top_k_results
        
        # Generate query embeddings
        query_dense = self.embeddings.embed_query(query)
        query_sparse = self._generate_sparse_vectors([query])[0]
        
        # Perform hybrid search
        logger.info(f"Performing hybrid search for query: {query[:50]}...")
        
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=query_dense,
                    using="dense",
                    limit=top_k * 2,
                ),
                Prefetch(
                    query=query_sparse,
                    using="sparse",
                    limit=top_k * 2,
                ),
            ],
            query=Query(
                fusion="rrf",  # Reciprocal Rank Fusion
            ),
            limit=top_k,
        )
        
        # Format results
        results = []
        for point in search_results.points:
            results.append({
                "id": point.id,
                "score": point.score,
                "text": point.payload.get("text", ""),
                "metadata": {
                    k: v for k, v in point.payload.items() if k != "text"
                },
            })
        
        logger.info(f"Found {len(results)} results")
        return results
    
    def delete_collection(self):
        """Delete the entire collection."""
        logger.warning(f"Deleting collection: {self.collection_name}")
        self.client.delete_collection(collection_name=self.collection_name)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        info = self.client.get_collection(collection_name=self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": info.status,
        }
