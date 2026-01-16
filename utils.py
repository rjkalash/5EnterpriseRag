"""
Utility functions for the RAG system.
"""
from typing import List, Dict, Any
import tiktoken
from loguru import logger


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text for a given model.
    
    Args:
        text: Text to count tokens for
        model: Model name
        
    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Error counting tokens: {e}")
        # Fallback: rough estimate
        return len(text) // 4


def truncate_to_token_limit(
    text: str,
    max_tokens: int,
    model: str = "gpt-4"
) -> str:
    """
    Truncate text to fit within token limit.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        model: Model name
        
    Returns:
        Truncated text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    
    except Exception as e:
        logger.warning(f"Error truncating text: {e}")
        # Fallback: character-based truncation
        char_limit = max_tokens * 4
        return text[:char_limit]


def format_sources(contexts: List[Dict[str, Any]]) -> str:
    """
    Format context sources for citation.
    
    Args:
        contexts: List of context dictionaries
        
    Returns:
        Formatted source string
    """
    sources = []
    seen = set()
    
    for ctx in contexts:
        metadata = ctx.get("metadata", {})
        source = metadata.get("source", "Unknown")
        
        if source not in seen:
            sources.append(source)
            seen.add(source)
    
    if not sources:
        return "No sources available"
    
    return "Sources: " + ", ".join(sources)


def calculate_relevance_score(
    query: str,
    text: str,
    method: str = "simple"
) -> float:
    """
    Calculate relevance score between query and text.
    
    Args:
        query: Query string
        text: Text to compare
        method: Scoring method ('simple' or 'advanced')
        
    Returns:
        Relevance score between 0 and 1
    """
    if method == "simple":
        # Simple word overlap
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & text_words)
        return overlap / len(query_words)
    
    else:
        # Could implement more advanced scoring here
        return 0.0


def merge_contexts(
    contexts: List[Dict[str, Any]],
    max_length: int = 4000
) -> str:
    """
    Merge multiple contexts into a single string.
    
    Args:
        contexts: List of context dictionaries
        max_length: Maximum character length
        
    Returns:
        Merged context string
    """
    merged = []
    current_length = 0
    
    for ctx in contexts:
        text = ctx.get("text", "")
        
        if current_length + len(text) > max_length:
            # Truncate last context to fit
            remaining = max_length - current_length
            if remaining > 100:  # Only add if meaningful
                merged.append(text[:remaining] + "...")
            break
        
        merged.append(text)
        current_length += len(text)
    
    return "\n\n".join(merged)


def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """
    Extract top keywords from text.
    
    Args:
        text: Input text
        top_k: Number of keywords to extract
        
    Returns:
        List of keywords
    """
    # Simple frequency-based extraction
    words = text.lower().split()
    
    # Filter out common words (simple stopwords)
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
        'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had'
    }
    
    filtered_words = [w for w in words if w not in stopwords and len(w) > 3]
    
    # Count frequencies
    word_freq = {}
    for word in filtered_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    return [word for word, _ in sorted_words[:top_k]]


def create_summary_stats(contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create summary statistics for retrieved contexts.
    
    Args:
        contexts: List of context dictionaries
        
    Returns:
        Dictionary with statistics
    """
    if not contexts:
        return {
            "total_contexts": 0,
            "avg_score": 0.0,
            "total_chars": 0,
            "sources": []
        }
    
    scores = [ctx.get("score", 0.0) for ctx in contexts]
    texts = [ctx.get("text", "") for ctx in contexts]
    sources = list(set(
        ctx.get("metadata", {}).get("source", "Unknown")
        for ctx in contexts
    ))
    
    return {
        "total_contexts": len(contexts),
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "total_chars": sum(len(t) for t in texts),
        "avg_chars": sum(len(t) for t in texts) / len(texts) if texts else 0,
        "sources": sources,
        "unique_sources": len(sources)
    }
