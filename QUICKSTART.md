# Quick Start Guide

Get your Enterprise RAG Knowledge Base running in 5 minutes!

## Prerequisites

- Python 3.9+
- Docker Desktop
- OpenAI API Key

## Step 1: Setup (2 minutes)

```bash
# Navigate to project
cd enterprise-rag-kb

# Run automated setup
python setup.py
```

This will:
- ✓ Check Python version
- ✓ Verify Docker installation
- ✓ Create .env file
- ✓ Install dependencies
- ✓ Start Qdrant
- ✓ Create necessary directories

## Step 2: Configure (1 minute)

Edit `.env` file and add your OpenAI API key:

```bash
OPENAI_API_KEY=sk-your-actual-key-here
```

## Step 3: Start Server (30 seconds)

```bash
python main.py
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

## Step 4: Test It! (1 minute)

### Option A: Web Browser

Visit: http://localhost:8000/docs

Try the interactive API documentation!

### Option B: Python

```python
import requests

# Ingest a document
requests.post(
    "http://localhost:8000/ingest",
    json={
        "texts": ["Python is a programming language known for simplicity."],
        "metadatas": [{"source": "test.txt"}]
    }
)

# Query it
response = requests.post(
    "http://localhost:8000/query",
    json={"question": "What is Python known for?"}
)

print(response.json()["answer"])
```

### Option C: cURL

```bash
# Ingest
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"texts": ["AI is artificial intelligence"]}'

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is AI?"}'
```

## Step 5: Try Examples (1 minute)

```bash
python examples.py
```

This runs 5 comprehensive examples demonstrating all features.

## What's Next?

### Upload Real Documents

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@your-document.pdf"
```

### Explore the Playground

Visit: http://localhost:8000/langserve/playground

Interactive interface for testing queries!

### Check System Health

```bash
curl http://localhost:8000/health
```

### View Collection Info

```bash
curl http://localhost:8000/collection/info
```

## Common Issues

### Issue: "Docker not found"

**Solution:** Install Docker Desktop from https://www.docker.com/

### Issue: "OpenAI API key not set"

**Solution:** Edit `.env` and add your key:
```
OPENAI_API_KEY=sk-your-key-here
```

### Issue: "Port 6333 already in use"

**Solution:** Stop existing Qdrant:
```bash
docker stop qdrant
docker rm qdrant
```

### Issue: "Module not found"

**Solution:** Reinstall dependencies:
```bash
pip install -r requirements.txt
```

## Quick Commands Reference

```bash
# Start Qdrant manually
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Stop Qdrant
docker stop qdrant

# Start API
python main.py

# Run tests
python test_rag.py

# Run examples
python examples.py

# View logs
tail -f logs/app.log  # if logging to file
```

## Architecture at a Glance

```
Your Question
     ↓
  FastAPI
     ↓
  RAG Chain
     ↓
Hybrid Search (Qdrant)
     ↓
  GPT-4
     ↓
  Answer
```

## Key Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/health` | Check system status |
| `/query` | Ask questions |
| `/ingest` | Add documents |
| `/upload` | Upload files |
| `/evaluate` | Measure performance |
| `/docs` | API documentation |
| `/langserve/playground` | Interactive testing |

## Performance Tips

1. **Batch queries** for multiple questions
2. **Adjust top_k** (3-10) based on needs
3. **Use metadata filters** for targeted search
4. **Monitor evaluation metrics** regularly
5. **Tune hybrid weights** for your domain

## Next Steps

1. ✅ Read `README.md` for detailed documentation
2. ✅ Check `API_DOCS.md` for API reference
3. ✅ Review `examples.py` for code samples
4. ✅ Read `PROJECT_SUMMARY.md` for architecture
5. ✅ Start building your knowledge base!

## Support

- 📖 Documentation: README.md, API_DOCS.md
- 💻 Examples: examples.py
- 🐛 Issues: GitHub Issues
- 📧 Questions: Open a discussion

---

**You're all set! Start building amazing RAG applications! 🚀**
