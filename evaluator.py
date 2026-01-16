"""
Ragas evaluation framework integration for RAG system quality assessment.
"""
from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevancy,
)
from loguru import logger

from config import settings


class RAGEvaluator:
    """Evaluates RAG system performance using Ragas metrics."""
    
    def __init__(self):
        """Initialize evaluator with Ragas metrics."""
        self.metrics = [
            faithfulness,           # Answer faithfulness to context
            answer_relevancy,       # Answer relevance to question
            context_precision,      # Precision of retrieved contexts
            context_recall,         # Recall of retrieved contexts
            context_relevancy,      # Relevance of contexts to question
        ]
    
    def prepare_evaluation_data(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None,
    ) -> Dataset:
        """
        Prepare data for Ragas evaluation.
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of context lists (each question has multiple contexts)
            ground_truths: Optional ground truth answers
            
        Returns:
            Hugging Face Dataset for evaluation
        """
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        
        if ground_truths:
            data["ground_truth"] = ground_truths
        
        return Dataset.from_dict(data)
    
    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate RAG system performance.
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of context lists
            ground_truths: Optional ground truth answers
            
        Returns:
            Dictionary of metric scores
        """
        logger.info(f"Evaluating {len(questions)} Q&A pairs")
        
        # Prepare dataset
        dataset = self.prepare_evaluation_data(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths,
        )
        
        # Select metrics based on available data
        metrics_to_use = self.metrics.copy()
        if not ground_truths:
            # Remove metrics that require ground truth
            metrics_to_use = [
                m for m in metrics_to_use 
                if m not in [context_recall]
            ]
        
        # Run evaluation
        logger.info("Running Ragas evaluation...")
        results = evaluate(
            dataset=dataset,
            metrics=metrics_to_use,
        )
        
        # Extract scores
        scores = {
            metric: results[metric] 
            for metric in results.keys()
        }
        
        logger.info("Evaluation complete")
        self._log_scores(scores)
        
        return scores
    
    def _log_scores(self, scores: Dict[str, float]):
        """Log evaluation scores."""
        logger.info("=" * 50)
        logger.info("RAGAS EVALUATION RESULTS")
        logger.info("=" * 50)
        
        for metric, score in scores.items():
            logger.info(f"{metric}: {score:.4f}")
        
        logger.info("=" * 50)
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = None,
    ) -> Dict[str, float]:
        """
        Evaluate a single Q&A pair.
        
        Args:
            question: Question
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Optional ground truth answer
            
        Returns:
            Dictionary of metric scores
        """
        return self.evaluate(
            questions=[question],
            answers=[answer],
            contexts=[contexts],
            ground_truths=[ground_truth] if ground_truth else None,
        )
    
    def compare_systems(
        self,
        questions: List[str],
        system_outputs: Dict[str, List[Dict[str, Any]]],
        ground_truths: List[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple RAG system configurations.
        
        Args:
            questions: List of questions
            system_outputs: Dict mapping system name to outputs
            ground_truths: Optional ground truth answers
            
        Returns:
            Dictionary mapping system name to scores
        """
        logger.info(f"Comparing {len(system_outputs)} systems")
        
        comparison_results = {}
        
        for system_name, outputs in system_outputs.items():
            logger.info(f"Evaluating system: {system_name}")
            
            answers = [out["answer"] for out in outputs]
            contexts = [
                [ctx["text"] for ctx in out["contexts"]]
                for out in outputs
            ]
            
            scores = self.evaluate(
                questions=questions,
                answers=answers,
                contexts=contexts,
                ground_truths=ground_truths,
            )
            
            comparison_results[system_name] = scores
        
        self._log_comparison(comparison_results)
        
        return comparison_results
    
    def _log_comparison(self, results: Dict[str, Dict[str, float]]):
        """Log comparison results."""
        logger.info("=" * 70)
        logger.info("SYSTEM COMPARISON")
        logger.info("=" * 70)
        
        # Get all metrics
        all_metrics = set()
        for scores in results.values():
            all_metrics.update(scores.keys())
        
        # Print header
        systems = list(results.keys())
        header = f"{'Metric':<25} " + " ".join(f"{s:<15}" for s in systems)
        logger.info(header)
        logger.info("-" * 70)
        
        # Print scores
        for metric in sorted(all_metrics):
            row = f"{metric:<25} "
            for system in systems:
                score = results[system].get(metric, 0.0)
                row += f"{score:<15.4f} "
            logger.info(row)
        
        logger.info("=" * 70)
    
    def calculate_improvement(
        self,
        baseline_scores: Dict[str, float],
        improved_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate percentage improvement over baseline.
        
        Args:
            baseline_scores: Baseline system scores
            improved_scores: Improved system scores
            
        Returns:
            Dictionary of improvement percentages
        """
        improvements = {}
        
        for metric in baseline_scores.keys():
            if metric in improved_scores:
                baseline = baseline_scores[metric]
                improved = improved_scores[metric]
                
                if baseline > 0:
                    improvement = ((improved - baseline) / baseline) * 100
                    improvements[metric] = improvement
        
        logger.info("Improvement Analysis:")
        for metric, improvement in improvements.items():
            logger.info(f"{metric}: {improvement:+.2f}%")
        
        return improvements
