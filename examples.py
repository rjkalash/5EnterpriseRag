"""
Example usage of the Enterprise RAG Knowledge Base.
"""
import asyncio
from loguru import logger

from vector_store import QdrantVectorStore
from rag_chain import RAGChain
from evaluator import RAGEvaluator
from document_processor import DocumentProcessor


async def example_basic_usage():
    """Basic usage example: ingest and query."""
    logger.info("=" * 50)
    logger.info("EXAMPLE 1: Basic Usage")
    logger.info("=" * 50)
    
    # Initialize components
    vector_store = QdrantVectorStore()
    rag_chain = RAGChain(vector_store)
    
    # Sample documents
    documents = [
        """
        Artificial Intelligence (AI) is the simulation of human intelligence by machines.
        It includes learning, reasoning, and self-correction. Common applications include
        natural language processing, computer vision, and robotics.
        """,
        """
        Machine Learning is a subset of AI that enables systems to learn from data
        without explicit programming. It uses algorithms to identify patterns and
        make predictions. Popular techniques include supervised learning, unsupervised
        learning, and reinforcement learning.
        """,
        """
        Deep Learning is a subset of machine learning based on artificial neural networks.
        It excels at processing unstructured data like images, audio, and text.
        Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) are
        common architectures.
        """,
    ]
    
    metadatas = [
        {"source": "AI_Basics.pdf", "topic": "artificial_intelligence"},
        {"source": "ML_Guide.pdf", "topic": "machine_learning"},
        {"source": "DL_Overview.pdf", "topic": "deep_learning"},
    ]
    
    # Ingest documents
    logger.info("Ingesting documents...")
    doc_ids = vector_store.add_documents(documents, metadatas)
    logger.info(f"Ingested {len(doc_ids)} document chunks")
    
    # Query the system
    question = "What is the difference between machine learning and deep learning?"
    logger.info(f"\nQuestion: {question}")
    
    result = rag_chain.query(
        question=question,
        return_contexts=True,
    )
    
    logger.info(f"\nAnswer: {result['answer']}")
    logger.info(f"\nRetrieved {len(result['contexts'])} contexts")


async def example_file_processing():
    """Example: Process files from a directory."""
    logger.info("=" * 50)
    logger.info("EXAMPLE 2: File Processing")
    logger.info("=" * 50)
    
    vector_store = QdrantVectorStore()
    doc_processor = DocumentProcessor()
    
    # Create sample text file
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("""
        Retrieval-Augmented Generation (RAG) combines information retrieval with
        text generation. It retrieves relevant documents from a knowledge base
        and uses them as context for generating accurate, grounded responses.
        This approach significantly reduces hallucinations in language models.
        """)
        temp_file = f.name
    
    try:
        # Process file
        logger.info(f"Processing file: {temp_file}")
        documents = doc_processor.load_file(temp_file)
        
        # Extract and ingest
        texts, metadatas = doc_processor.extract_text_and_metadata(documents)
        doc_ids = vector_store.add_documents(texts, metadatas)
        
        logger.info(f"Successfully ingested {len(doc_ids)} chunks")
        
    finally:
        os.unlink(temp_file)


async def example_evaluation():
    """Example: Evaluate RAG system performance."""
    logger.info("=" * 50)
    logger.info("EXAMPLE 3: RAG Evaluation with Ragas")
    logger.info("=" * 50)
    
    vector_store = QdrantVectorStore()
    rag_chain = RAGChain(vector_store)
    evaluator = RAGEvaluator()
    
    # Sample documents
    documents = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "JavaScript is primarily used for web development and runs in browsers.",
        "Java is a statically-typed language commonly used for enterprise applications.",
    ]
    
    vector_store.add_documents(documents)
    
    # Test questions
    questions = [
        "What is Python known for?",
        "Where does JavaScript run?",
        "What type of language is Java?",
    ]
    
    # Ground truth answers (optional)
    ground_truths = [
        "Python is known for its simplicity and readability.",
        "JavaScript runs in browsers.",
        "Java is a statically-typed language.",
    ]
    
    # Generate answers
    logger.info("Generating answers...")
    results = rag_chain.batch_query(questions)
    
    answers = [r["answer"] for r in results]
    contexts = [[ctx["text"] for ctx in r["contexts"]] for r in results]
    
    # Evaluate
    logger.info("Running evaluation...")
    scores = evaluator.evaluate(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths,
    )
    
    logger.info("\nEvaluation complete!")
    logger.info(f"Average faithfulness: {scores.get('faithfulness', 0):.4f}")
    logger.info(f"Average answer relevancy: {scores.get('answer_relevancy', 0):.4f}")


async def example_hybrid_search():
    """Example: Demonstrate hybrid search capabilities."""
    logger.info("=" * 50)
    logger.info("EXAMPLE 4: Hybrid Search (Dense + Sparse)")
    logger.info("=" * 50)
    
    vector_store = QdrantVectorStore()
    
    # Technical documents with specific terminology
    documents = [
        """
        Kubernetes is an open-source container orchestration platform. It automates
        deployment, scaling, and management of containerized applications. Key concepts
        include pods, services, deployments, and namespaces.
        """,
        """
        Docker is a platform for developing, shipping, and running applications in
        containers. Containers package software with all dependencies, ensuring
        consistency across environments. Docker uses images and containers as core concepts.
        """,
        """
        Microservices architecture breaks applications into small, independent services.
        Each service runs in its own process and communicates via APIs. This approach
        improves scalability, flexibility, and fault isolation.
        """,
    ]
    
    metadatas = [
        {"source": "k8s_guide.pdf", "category": "orchestration"},
        {"source": "docker_intro.pdf", "category": "containerization"},
        {"source": "microservices.pdf", "category": "architecture"},
    ]
    
    vector_store.add_documents(documents, metadatas)
    
    # Test hybrid search
    query = "How do you manage containerized applications at scale?"
    
    logger.info(f"Query: {query}")
    logger.info("\nPerforming hybrid search (BM25 + Dense Embeddings)...")
    
    results = vector_store.hybrid_search(query, top_k=3)
    
    logger.info(f"\nFound {len(results)} results:")
    for idx, result in enumerate(results, 1):
        logger.info(f"\n[Result {idx}] (Score: {result['score']:.4f})")
        logger.info(f"Source: {result['metadata'].get('source', 'Unknown')}")
        logger.info(f"Text: {result['text'][:150]}...")


async def example_batch_processing():
    """Example: Batch processing multiple questions."""
    logger.info("=" * 50)
    logger.info("EXAMPLE 5: Batch Query Processing")
    logger.info("=" * 50)
    
    vector_store = QdrantVectorStore()
    rag_chain = RAGChain(vector_store)
    
    # Knowledge base
    documents = [
        "The capital of France is Paris, known for the Eiffel Tower.",
        "Tokyo is the capital of Japan and the world's most populous metropolitan area.",
        "London is the capital of the United Kingdom and a major financial center.",
        "Berlin is the capital of Germany, known for its history and culture.",
    ]
    
    vector_store.add_documents(documents)
    
    # Multiple questions
    questions = [
        "What is the capital of France?",
        "Which city is the capital of Japan?",
        "Tell me about London.",
        "What is Berlin known for?",
    ]
    
    logger.info(f"Processing {len(questions)} questions in batch...")
    
    results = rag_chain.batch_query(questions, top_k=2)
    
    for idx, (question, result) in enumerate(zip(questions, results), 1):
        logger.info(f"\n[Q{idx}] {question}")
        logger.info(f"[A{idx}] {result['answer']}")


async def main():
    """Run all examples."""
    try:
        await example_basic_usage()
        await example_file_processing()
        await example_hybrid_search()
        await example_batch_processing()
        
        # Note: Evaluation example requires more setup and API calls
        # Uncomment to run:
        # await example_evaluation()
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
