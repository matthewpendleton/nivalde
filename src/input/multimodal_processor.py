import torch
import torch.nn as nn
import sounddevice as sd
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModel
from ..models.contextual_processor import ContextualProcessor

@dataclass
class MultimodalInput:
    audio_embedding: Optional[torch.Tensor]
    video_embedding: Optional[torch.Tensor]
    text_embedding: Optional[torch.Tensor]
    contextual_embedding: Optional[torch.Tensor] = None

class AudioEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.conv1d = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=32, nhead=8), num_layers=3
        )
        self.projection = nn.Linear(32, embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1d(x.unsqueeze(1))
        x = x.permute(2, 0, 1)  # (seq_len, batch, features)
        x = self.transformer(x)
        x = self.projection(x.mean(0))  # Average pooling over sequence
        return x

class TextEncoder:
    def __init__(self, model_name: str = "bhadresh-savani/bert-base-go-emotion"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        
    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling - take attention mask into account for correct averaging"""
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
        
    def encode(self, text: str, max_length: int = 512) -> torch.Tensor:
        """Encode text into emotion-aware embeddings"""
        # Tokenize and prepare input
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling with attention masking
            embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
        return embeddings

class MultimodalProcessor:
    def __init__(self, audio_embedding_dim: int = 256, contextual_dim: int = 768):
        self.audio_encoder = AudioEncoder(input_dim=1024, embedding_dim=audio_embedding_dim)
        self.text_encoder = TextEncoder()
        self.contextual_processor = ContextualProcessor(embedding_dim=contextual_dim)
        self.audio_buffer = []
        self.is_recording = False
        
    def start_audio_stream(self, sample_rate: int = 44100):
        """Start real-time audio processing"""
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            self.audio_buffer.extend(indata.flatten())
            
            # Process buffer when it reaches sufficient size
            if len(self.audio_buffer) >= sample_rate:  # Process 1 second chunks
                audio_data = torch.tensor(self.audio_buffer[:sample_rate])
                embedding = self.process_audio(audio_data)
                self.audio_buffer = self.audio_buffer[sample_rate:]
                
        self.audio_stream = sd.InputStream(
            channels=1,
            samplerate=sample_rate,
            callback=audio_callback
        )
        self.audio_stream.start()
        self.is_recording = True
        
    def stop_audio_stream(self):
        """Stop audio processing"""
        if self.is_recording:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.is_recording = False
            
    def process_audio(self, audio_data: torch.Tensor) -> torch.Tensor:
        """Process raw audio directly to embeddings"""
        with torch.no_grad():
            return self.audio_encoder(audio_data)
    
    def process_text(self, text: str) -> torch.Tensor:
        """Process text input to embeddings"""
        return self.text_encoder.encode(text)
            
    def process_input(self, 
                     audio_data: Optional[torch.Tensor] = None,
                     video_frame: Optional[np.ndarray] = None,
                     text: Optional[str] = None) -> MultimodalInput:
        """Process multimodal input and return embeddings with contextual understanding"""
        # Process available modalities
        audio_embedding = self.process_audio(audio_data) if audio_data is not None else None
        text_embedding = self.process_text(text) if text is not None else None
        
        # Get contextual embedding using BERT
        contextual_embedding = self.contextual_processor.process(
            audio_embedding=audio_embedding,
            video_embedding=None,  # Placeholder for video processing
            text_embedding=text_embedding
        )
        
        return MultimodalInput(
            audio_embedding=audio_embedding,
            video_embedding=None,  # Placeholder for video processing
            text_embedding=text_embedding,
            contextual_embedding=contextual_embedding
        )
