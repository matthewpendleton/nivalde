"""BERT-based input contextualization.

This module handles the contextualization of client input using BERT,
providing emotional and semantic context to the system.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Union

class BERTContextualizer(nn.Module):
    """Contextualizes input using BERT with emotional emphasis."""
    
    def __init__(self, model_name: str = "bert-large-uncased"):
        """Initialize the BERT contextualizer.
        
        Args:
            model_name: Name of the BERT model to use
        """
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Emotional emphasis layer
        hidden_size = self.bert.config.hidden_size
        self.emotional_emphasis = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, 
                text: Union[str, List[str]],
                emotional_keywords: List[str] = None) -> torch.Tensor:
        """Contextualize input text with emotional emphasis.
        
        Args:
            text: Input text or list of texts
            emotional_keywords: Optional list of emotional keywords to emphasize
            
        Returns:
            Contextualized representation
        """
        # Prepare input
        if isinstance(text, str):
            text = [text]
            
        # Tokenize with attention to emotional keywords
        tokenizer_output = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Get BERT embeddings
        with torch.no_grad():
            bert_output = self.bert(**tokenizer_output)
            
        # Get CLS token embedding
        context = bert_output.last_hidden_state[:, 0]
        
        # Apply emotional emphasis if keywords provided
        if emotional_keywords:
            # Tokenize emotional keywords
            keyword_tokens = self.tokenizer(
                emotional_keywords,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Get keyword embeddings
            with torch.no_grad():
                keyword_output = self.bert(**keyword_tokens)
            keyword_embeds = keyword_output.last_hidden_state[:, 0]
            
            # Compute attention with keywords
            attention = torch.matmul(
                context,
                keyword_embeds.transpose(-2, -1)
            ) / torch.sqrt(torch.tensor(context.size(-1)))
            
            # Weight context with keyword attention
            context = context * (1 + torch.softmax(attention, dim=-1).mean(dim=-1, keepdim=True))
            
        # Apply emotional emphasis layer
        context = self.emotional_emphasis(context)
        
        return context[0] if len(text) == 1 else context
        
    def batch_contextualize(self,
                           texts: List[str],
                           metadata: List[Dict] = None) -> torch.Tensor:
        """Contextualize a batch of texts with optional metadata.
        
        Args:
            texts: List of input texts
            metadata: Optional list of metadata dicts with emotional_keywords
            
        Returns:
            Batch of contextualized representations
        """
        if metadata is None:
            return self.forward(texts)
            
        # Process each text with its metadata
        contexts = []
        for text, meta in zip(texts, metadata):
            context = self.forward(
                text,
                emotional_keywords=meta.get('emotional_keywords')
            )
            contexts.append(context)
            
        return torch.stack(contexts)
