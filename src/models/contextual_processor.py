import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from typing import Optional, Tuple

class MultimodalBertEncoder(nn.Module):
    def __init__(self, 
                 embedding_dim: int = 768,
                 num_layers: int = 6,
                 num_heads: int = 8):
        super().__init__()
        
        # Custom BERT configuration with fewer layers for real-time processing
        self.config = BertConfig(
            hidden_size=embedding_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=embedding_dim * 4,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=3  # For different modality types
        )
        
        # Initialize BERT with custom config
        self.bert = BertModel(self.config)
        
        # Modality-specific projection layers
        self.audio_projection = nn.Linear(256, embedding_dim)  # Assuming audio_dim=256
        self.video_projection = nn.Linear(512, embedding_dim)  # Assuming video_dim=512
        self.text_projection = nn.Linear(768, embedding_dim)   # Back to 768 for emotion BERT
        
        # Modality type embeddings
        self.modality_embeddings = nn.Embedding(3, embedding_dim)
        
    def project_inputs(self,
                      audio_embedding: Optional[torch.Tensor] = None,
                      video_embedding: Optional[torch.Tensor] = None,
                      text_embedding: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project different modalities to common embedding space"""
        batch_size = max(
            x.size(0) if x is not None else 0 
            for x in [audio_embedding, video_embedding, text_embedding]
        )
        
        # Initialize sequence and attention mask
        sequence = []
        modality_ids = []
        attention_mask = []
        
        # Process available modalities
        if audio_embedding is not None:
            audio_proj = self.audio_projection(audio_embedding)
            sequence.append(audio_proj)
            modality_ids.extend([0] * audio_proj.size(0))
            attention_mask.extend([1] * audio_proj.size(0))
            
        if video_embedding is not None:
            video_proj = self.video_projection(video_embedding)
            sequence.append(video_proj)
            modality_ids.extend([1] * video_proj.size(0))
            attention_mask.extend([1] * video_proj.size(0))
            
        if text_embedding is not None:
            text_proj = self.text_projection(text_embedding)
            sequence.append(text_proj)
            modality_ids.extend([2] * text_proj.size(0))
            attention_mask.extend([1] * text_proj.size(0))
        
        # Handle case where no modalities are present
        if not sequence:
            return torch.zeros((1, self.config.hidden_size)), torch.zeros((1,))
            
        # Concatenate all modalities
        sequence = torch.cat(sequence, dim=0)
        modality_ids = torch.tensor(modality_ids, device=sequence.device)
        attention_mask = torch.tensor(attention_mask, device=sequence.device)
        
        return sequence, attention_mask, modality_ids
        
    def forward(self,
                audio_embedding: Optional[torch.Tensor] = None,
                video_embedding: Optional[torch.Tensor] = None,
                text_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process multimodal input through BERT"""
        # Project inputs to common space
        sequence, attention_mask, modality_ids = self.project_inputs(
            audio_embedding, video_embedding, text_embedding
        )
        
        # Add modality embeddings
        sequence = sequence + self.modality_embeddings(modality_ids)
        
        # Process through BERT
        outputs = self.bert(
            inputs_embeds=sequence.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0)
        )
        
        # Return pooled output (CLS token)
        return outputs.pooler_output

class ContextualProcessor:
    def __init__(self, embedding_dim: int = 768):
        self.encoder = MultimodalBertEncoder(embedding_dim=embedding_dim)
        
    def process(self,
                audio_embedding: Optional[torch.Tensor] = None,
                video_embedding: Optional[torch.Tensor] = None,
                text_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process multimodal input and return contextual embedding"""
        with torch.no_grad():
            return self.encoder(
                audio_embedding=audio_embedding,
                video_embedding=video_embedding,
                text_embedding=text_embedding
            )
