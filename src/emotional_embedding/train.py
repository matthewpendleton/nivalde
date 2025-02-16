"""Training script for the Emotional Embedding Space."""

import torch
import torch.optim as optim
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
import json
import glob
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple

from src.emotional_embedding.ees import EmotionalEmbeddingSpace
from src.memory.memory_system import TransformerMemorySystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_therapy_sessions(data_dir: str) -> List[Dict]:
    """Load therapy sessions from generated JSON files.
    
    Args:
        data_dir: Directory containing generated session files
        
    Returns:
        List of session dictionaries
    """
    sessions = []
    for filepath in tqdm(glob.glob(f"{data_dir}/client_*.json"), desc="Loading sessions"):
        with open(filepath, 'r') as f:
            client_sessions = json.load(f)
            sessions.extend(client_sessions)
    return sessions

def preprocess_sessions(
    sessions: List[Dict],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str
) -> List[List[torch.Tensor]]:
    """Preprocess therapy sessions into BERT embeddings.
    
    Args:
        sessions: List of session dictionaries
        tokenizer: BERT tokenizer
        model: BERT model
        device: Device to compute embeddings on
        
    Returns:
        List of embedded dialogue sequences
    """
    embedded_sequences = []
    
    for session in tqdm(sessions, desc="Processing sessions"):
        sequence = []
        for utterance in session['dialog']:
            # Tokenize and get BERT embedding
            inputs = tokenizer(
                utterance,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :]
                sequence.append(embedding)
        
        embedded_sequences.append(sequence)
    
    return embedded_sequences

def train_ees(
    train_sequences,
    val_sequences=None,
    input_dim=768,
    latent_dim=32,
    hidden_dim=128,
    batch_size=16,
    num_epochs=100,
    learning_rate=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """Train the Emotional Embedding Space model.
    
    Args:
        train_sequences: List of dialogue sequences for training
        val_sequences: Optional validation sequences
        input_dim: Input dimension size
        latent_dim: Latent space dimension size
        hidden_dim: Hidden layer dimension size
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        device: Device to train on
        
    Returns:
        Trained EES model
    """
    # Initialize models
    ees = EmotionalEmbeddingSpace(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim
    ).to(device)
    
    memory_system = TransformerMemorySystem(
        embedding_dim=input_dim
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(
        list(ees.parameters()) + list(memory_system.parameters()),
        lr=learning_rate
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        ees.train()
        memory_system.train()
        
        # Shuffle training sequences
        np.random.shuffle(train_sequences)
        
        # Process sequences in batches
        total_loss = 0
        num_batches = 0
        
        for i in tqdm(range(0, len(train_sequences), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_sequences = train_sequences[i:i + batch_size]
            batch_loss = 0
            
            optimizer.zero_grad()
            
            # Process each sequence
            for sequence in batch_sequences:
                # Reset state for new sequence
                ees.previous_state = None
                memory_context = torch.zeros(1, input_dim).to(device)
                
                # Process each utterance
                for utterance_embedding in sequence:
                    # Get BERT context (using same embedding for now)
                    bert_context = utterance_embedding
                    
                    # Compute loss
                    loss = ees.compute_loss(
                        utterance_embedding,
                        bert_context,
                        memory_context
                    )
                    batch_loss += loss
                    
                    # Update memory
                    memory_context = memory_system(
                        utterance_embedding,
                        memory_context
                    )
            
            # Average loss over batch
            batch_loss = batch_loss / (len(batch_sequences) * len(batch_sequences[0]))
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item()
            num_batches += 1
            
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Validation
        if val_sequences:
            val_loss = validate_ees(ees, memory_system, val_sequences, device)
            logger.info(f"Validation Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save checkpoint
                torch.save({
                    'ees_state_dict': ees.state_dict(),
                    'memory_state_dict': memory_system.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss
                }, 'checkpoints/best_model.pt')
    
    return ees, memory_system

def validate_ees(ees, memory_system, val_sequences, device):
    """Validate the EES model on validation sequences."""
    ees.eval()
    memory_system.eval()
    total_loss = 0
    num_sequences = len(val_sequences)
    
    with torch.no_grad():
        for sequence in val_sequences:
            ees.previous_state = None
            memory_context = torch.zeros(1, ees.input_dim).to(device)
            sequence_loss = 0
            
            for utterance_embedding in sequence:
                bert_context = utterance_embedding
                loss = ees.compute_loss(
                    utterance_embedding,
                    bert_context,
                    memory_context
                )
                sequence_loss += loss
                memory_context = memory_system(
                    utterance_embedding,
                    memory_context
                )
            
            total_loss += sequence_loss / len(sequence)
    
    return total_loss / num_sequences

if __name__ == "__main__":
    # Initialize BERT model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model = AutoModel.from_pretrained("bert-base-uncased").to(device)
    
    # Load and preprocess sessions
    logger.info("Loading therapy sessions...")
    sessions = load_therapy_sessions("data/generated_sessions")
    
    # Split into train/val
    np.random.shuffle(sessions)
    split_idx = int(len(sessions) * 0.9)  # 90% train, 10% val
    train_sessions = sessions[:split_idx]
    val_sessions = sessions[split_idx:]
    
    logger.info("Preprocessing sessions...")
    train_sequences = preprocess_sessions(train_sessions, tokenizer, bert_model, device)
    val_sequences = preprocess_sessions(val_sessions, tokenizer, bert_model, device)
    
    # Create checkpoints directory
    Path("checkpoints").mkdir(exist_ok=True)
    
    # Train model
    logger.info("Starting training...")
    ees, memory_system = train_ees(
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        device=device
    )
    
    logger.info("Training complete!")
