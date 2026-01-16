"""
RAG chain implementation using LangChain.
"""
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from loguru import logger

from config import settings
from vector_store import QdrantVectorStore


class RAGChain:
    """RAG chain for question answering with context retrieval."""
    
    def __init__(self, vector_store: QdrantVectorStore):
        """
        Initialize RAG chain.
        
        Args:
            vector_store: Qdrant vector store instance
        """
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.temperature,
            openai_api_key=settings.openai_api_key,
        )
        self.prompt = self._create_prompt()
        self.chain = self._create_chain()
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create the RAG prompt template."""
        template = """You are an expert assistant with access to a knowledge base. 
Use the following context to answer the question accurately and comprehensively.

If the context doesn't contain enough information to answer the question, say so clearly.
Always cite the relevant parts of the context in your answer.

Context:
{context}

Question: {question}

Answer:"""
        
        return ChatPromptTemplate.from_template(template)
    
    def _create_chain(self):
        """Create the RAG chain."""
        return (
            {
                "context": lambda x: self._format_context(x["contexts"]),
                "question": lambda x: x["question"],
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _format_context(self, contexts: List[Dict[str, Any]]) -> str:
        """
        Format retrieved contexts for the prompt.
        
        Args:
            contexts: List of retrieved context documents
            
        Returns:
            Formatted context string
        """
        formatted = []
        for idx, ctx in enumerate(contexts, 1):
            text = ctx.get("text", "")
            metadata = ctx.get("metadata", {})
            source = metadata.get("source", "Unknown")
            
            formatted.append(
                f"[Document {idx}] (Source: {source})\n{text}\n"
            )
        
        return "\n".join(formatted)
    
    def retrieve(
        self,
        question: str,
        top_k: int = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant contexts for a question.
        
        Args:
            question: User question
            top_k: Number of contexts to retrieve
            filter_conditions: Optional metadata filters
            
        Returns:
            List of retrieved contexts
        """
        logger.info(f"Retrieving contexts for: {question[:50]}...")
        
        contexts = self.vector_store.hybrid_search(
            query=question,
            top_k=top_k,
            filter_conditions=filter_conditions,
        )
        
        return contexts
    
    def generate(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
    ) -> str:
        """
        Generate answer from question and contexts.
        
        Args:
            question: User question
            contexts: Retrieved contexts
            
        Returns:
            Generated answer
        """
        logger.info("Generating answer...")
        
        answer = self.chain.invoke({
            "question": question,
            "contexts": contexts,
        })
        
        return answer
    
    def query(
        self,
        question: str,
        top_k: int = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        return_contexts: bool = False,
    ) -> Dict[str, Any]:
        """
        Complete RAG query: retrieve + generate.
        
        Args:
            question: User question
            top_k: Number of contexts to retrieve
            filter_conditions: Optional metadata filters
            return_contexts: Whether to return retrieved contexts
            
        Returns:
            Dictionary with answer and optionally contexts
        """
        # Retrieve contexts
        contexts = self.retrieve(
            question=question,
            top_k=top_k,
            filter_conditions=filter_conditions,
        )
        
        if not contexts:
            logger.warning("No contexts retrieved")
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "contexts": [] if return_contexts else None,
            }
        
        # Generate answer
        answer = self.generate(question=question, contexts=contexts)
        
        result = {"answer": answer}
        
        if return_contexts:
            result["contexts"] = contexts
        
        return result
    
    def batch_query(
        self,
        questions: List[str],
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple questions in batch.
        
        Args:
            questions: List of questions
            top_k: Number of contexts per question
            
        Returns:
            List of results
        """
        logger.info(f"Processing batch of {len(questions)} questions")
        
        results = []
        for question in questions:
            result = self.query(
                question=question,
                top_k=top_k,
                return_contexts=True,
            )
            results.append(result)
        
        return results
