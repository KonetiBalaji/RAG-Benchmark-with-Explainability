"""Embedding generation using OpenAI API.

Industry best practice: Batch processing with retry logic and cost tracking.
Reference: OpenAI embeddings documentation
"""

import time
from typing import List

import numpy as np
from openai import OpenAI
from loguru import logger
from tqdm import tqdm

from src.utils.config_loader import get_config
from src.utils.cost_tracker import get_cost_tracker


class EmbeddingGenerator:
    """Generate embeddings using OpenAI API.
    
    Best practices:
    - Batch processing for efficiency
    - Automatic retry with exponential backoff
    - Cost tracking
    - Rate limiting
    
    Reference: OpenAI Embeddings API
    https://platform.openai.com/docs/guides/embeddings
    """

    def __init__(self):
        """Initialize embedding generator."""
        self.config = get_config()
        self.cost_tracker = get_cost_tracker()
        
        # Get API key and initialize client
        import os
        api_key = self.config.get_api_key("openai")
        # Initialize client without organization to avoid mismatch errors
        # Remove OPENAI_ORG_ID from environment if set
        os.environ.pop('OPENAI_ORG_ID', None)
        self.client = OpenAI(api_key=api_key)
        
        # Configuration
        self.model = self.config.get("embeddings.model", "text-embedding-3-small")
        self.dimensions = self.config.get("embeddings.dimensions", 1536)
        self.batch_size = self.config.get("embeddings.batch_size", 100)
        self.max_retries = self.config.get("embeddings.max_retries", 3)
        
        logger.info(f"EmbeddingGenerator initialized: model={self.model}, dim={self.dimensions}")

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        embeddings = self.generate_embeddings([text])
        return embeddings[0]

    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[np.ndarray]:
        """Generate embeddings for multiple texts with batching.
        
        Args:
            texts: List of input texts
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
            
        Best practice: Process in batches to respect rate limits
        """
        if not texts:
            return []
        
        all_embeddings = []
        
        # Process in batches
        batches = [
            texts[i:i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        
        iterator = tqdm(batches, desc="Generating embeddings") if show_progress else batches
        
        for batch in iterator:
            batch_embeddings = self._generate_batch(batch)
            all_embeddings.extend(batch_embeddings)
        
        logger.info(f"Generated {len(all_embeddings)} embeddings")
        
        return all_embeddings

    def _generate_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch with retry logic.
        
        Args:
            texts: Batch of texts
            
        Returns:
            List of embeddings
        """
        for attempt in range(self.max_retries):
            try:
                # Call OpenAI API
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                    dimensions=self.dimensions,
                )
                
                # Extract embeddings
                embeddings = [np.array(item.embedding) for item in response.data]
                
                # Track cost
                total_tokens = response.usage.total_tokens
                self.cost_tracker.add_entry(
                    service="embeddings",
                    operation="generate",
                    tokens=total_tokens,
                    model=self.model,
                    metadata={"batch_size": len(texts)},
                )
                
                return embeddings
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        f"Embedding generation failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Embedding generation failed after {self.max_retries} attempts: {e}")
                    raise

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension.
        
        Returns:
            Embedding dimension
        """
        return self.dimensions
