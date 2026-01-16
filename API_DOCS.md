# API Documentation

## Overview

The Enterprise RAG Knowledge Base provides a comprehensive REST API for document ingestion, querying, and evaluation.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. In production, implement API key authentication or OAuth2.

## Endpoints

### Health Check

Check the system health and collection status.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "collection": {
    "name": "enterprise_knowledge_base",
    "points_count": 150,
    "vectors_count": 150,
    "status": "green"
  }
}
```

---

### Query Knowledge Base

Query the knowledge base with a question.

**Endpoint:** `POST /query`

**Request Body:**
```json
{
  "question": "What is machine learning?",
  "top_k": 5,
  "return_contexts": true
}
```

**Parameters:**
- `question` (required): The question to ask
- `top_k` (optional): Number of contexts to retrieve (default: 5)
- `return_contexts` (optional): Whether to return retrieved contexts (default: false)

**Response:**
```json
{
  "answer": "Machine learning is a subset of artificial intelligence...",
  "contexts": [
    {
      "id": "uuid-here",
      "score": 0.89,
      "text": "Machine learning is...",
      "metadata": {
        "source": "ml_guide.pdf",
        "chunk_index": 0
      }
    }
  ]
}
```

---

### Batch Query

Process multiple questions in a single request.

**Endpoint:** `POST /query/batch`

**Request Body:**
```json
{
  "questions": [
    "What is AI?",
    "What is ML?",
    "What is deep learning?"
  ],
  "top_k": 3
}
```

**Response:**
```json
{
  "results": [
    {
      "answer": "AI is...",
      "contexts": [...]
    },
    {
      "answer": "ML is...",
      "contexts": [...]
    }
  ],
  "count": 3
}
```

---

### Ingest Documents

Add documents to the knowledge base.

**Endpoint:** `POST /ingest`

**Request Body:**
```json
{
  "texts": [
    "Document text 1...",
    "Document text 2..."
  ],
  "metadatas": [
    {
      "source": "doc1.pdf",
      "topic": "AI",
      "author": "John Doe"
    },
    {
      "source": "doc2.pdf",
      "topic": "ML"
    }
  ]
}
```

**Response:**
```json
{
  "document_ids": ["uuid-1", "uuid-2", "uuid-3"],
  "count": 3
}
```

---

### Upload File

Upload and process a document file.

**Endpoint:** `POST /upload`

**Request:** Multipart form data with file

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "filename": "document.pdf",
  "document_count": 1,
  "chunk_count": 15,
  "document_ids": ["uuid-1", "uuid-2", ...]
}
```

**Supported Formats:**
- PDF (.pdf)
- Word (.docx)
- Text (.txt)

---

### Evaluate System

Evaluate RAG system performance using Ragas metrics.

**Endpoint:** `POST /evaluate`

**Request Body:**
```json
{
  "questions": [
    "What is Python?",
    "What is JavaScript?"
  ],
  "ground_truths": [
    "Python is a programming language",
    "JavaScript is used for web development"
  ]
}
```

**Response:**
```json
{
  "scores": {
    "faithfulness": 0.92,
    "answer_relevancy": 0.88,
    "context_precision": 0.85,
    "context_recall": 0.90,
    "context_relevancy": 0.87
  }
}
```

**Metrics Explained:**
- **Faithfulness**: How grounded the answer is in the context (0-1)
- **Answer Relevancy**: How relevant the answer is to the question (0-1)
- **Context Precision**: Precision of retrieved contexts (0-1)
- **Context Recall**: Recall of retrieved contexts (0-1, requires ground truth)
- **Context Relevancy**: Relevance of contexts to question (0-1)

---

### Collection Info

Get information about the vector collection.

**Endpoint:** `GET /collection/info`

**Response:**
```json
{
  "name": "enterprise_knowledge_base",
  "points_count": 150,
  "vectors_count": 150,
  "status": "green"
}
```

---

### Reset Collection

Delete and recreate the collection (use with caution!).

**Endpoint:** `DELETE /collection/reset`

**Response:**
```json
{
  "status": "Collection reset successfully"
}
```

---

## LangServe Endpoints

The application includes LangServe integration for advanced features.

### LangServe Playground

Interactive playground for testing the RAG chain.

**URL:** `http://localhost:8000/langserve/playground`

### LangServe Invoke

Directly invoke the RAG chain.

**Endpoint:** `POST /langserve/invoke`

**Request Body:**
```json
{
  "input": {
    "question": "What is AI?",
    "contexts": [...]
  }
}
```

---

## Error Responses

All endpoints return standard HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found
- `500`: Internal Server Error

**Error Response Format:**
```json
{
  "detail": "Error message here"
}
```

---

## Rate Limiting

Currently, no rate limiting is implemented. For production use, implement rate limiting using:
- FastAPI middleware
- Nginx/API Gateway
- Redis-based rate limiting

---

## Examples

### Python Client

```python
import requests

# Query
response = requests.post(
    "http://localhost:8000/query",
    json={
        "question": "What is machine learning?",
        "return_contexts": True
    }
)
result = response.json()
print(result["answer"])

# Upload file
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload",
        files={"file": f}
    )
print(response.json())
```

### JavaScript Client

```javascript
// Query
const response = await fetch('http://localhost:8000/query', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    question: 'What is machine learning?',
    return_contexts: true
  })
});

const result = await response.json();
console.log(result.answer);
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is AI?", "return_contexts": true}'

# Upload file
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf"

# Ingest documents
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Document 1", "Document 2"],
    "metadatas": [{"source": "doc1.txt"}, {"source": "doc2.txt"}]
  }'
```

---

## WebSocket Support

Currently not implemented. Future versions may include WebSocket support for:
- Real-time query streaming
- Progress updates for long-running operations
- Live evaluation metrics

---

## Monitoring

### Prometheus Metrics

Future versions will expose Prometheus metrics at `/metrics`:
- Request count
- Response time
- Error rate
- Vector store operations

### Logging

Logs are written to:
- Console (stdout)
- `logs/` directory (if configured)

Log levels: DEBUG, INFO, WARNING, ERROR

---

## Best Practices

1. **Chunking**: Keep document chunks between 500-1500 characters
2. **Metadata**: Always include source information in metadata
3. **Batch Operations**: Use batch endpoints for multiple queries
4. **Error Handling**: Always check response status codes
5. **Context Size**: Adjust `top_k` based on your use case (3-10 recommended)

---

## Support

For issues or questions:
- Check the main README.md
- Review examples.py
- Open an issue on GitHub
