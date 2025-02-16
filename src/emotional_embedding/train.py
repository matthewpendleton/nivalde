"""Training script for the Emotional Embedding Space."""

import torch
import torch.optim as optim
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
import json
import glob
import random
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Iterator
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.emotional_embedding.ees import EmotionalEmbeddingSpace
from src.memory.memory_system import TransformerMemorySystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TherapySessionDataset(Dataset):
    """Dataset for therapy sessions with batching support."""
    
    def __init__(
        self,
        data_dir: str,
        tokenizer: AutoTokenizer,
        model: AutoModel,
        device: str,
        max_sessions: int = None,
        validation: bool = False,
        val_ratio: float = 0.1,
        batch_size: int = 16,
        cache_size: int = 1000,  # Number of sessions to cache in memory
        checkpoint_dir: str = "data/checkpoints"
    ):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.max_sessions = max_sessions
        self.validation = validation
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.checkpoint_dir = checkpoint_dir
        
        # Get list of all client files
        self.files = sorted(glob.glob(f"{data_dir}/client_*.json"))
        if max_sessions:
            random.shuffle(self.files)
            num_files = int(max_sessions / 10)  # Each file has 10 sessions
            self.files = self.files[:num_files]
        
        # Split files for validation
        split_idx = int(len(self.files) * (1 - val_ratio))
        self.files = self.files[split_idx:] if validation else self.files[:split_idx]
        
        # Initialize cache
        self.cache = []
        self.cache_indices = set()
        
        # Try to load from checkpoint
        self._load_or_create_cache()
    
    def _load_or_create_cache(self):
        """Load cache from checkpoint if exists, otherwise create new cache."""
        dataset_type = "validation" if self.validation else "training"
        checkpoint_path = Path(self.checkpoint_dir) / f"dataset_cache_{dataset_type}.pt"
        
        if checkpoint_path.exists():
            logger.info(f"Loading {dataset_type} cache from checkpoint")
            try:
                checkpoint = torch.load(checkpoint_path)
                self.cache = checkpoint['cache']
                self.files = checkpoint['remaining_files']
                logger.info(f"Loaded {len(self.cache)} cached sessions")
                return
            except Exception as e:
                logger.error(f"Error loading checkpoint: {str(e)}")
        
        logger.info(f"No valid checkpoint found, creating new cache")
        self._fill_cache()
    
    def _save_checkpoint(self):
        """Save current cache state to checkpoint."""
        dataset_type = "validation" if self.validation else "training"
        checkpoint_path = Path(self.checkpoint_dir) / f"dataset_cache_{dataset_type}.pt"
        
        checkpoint = {
            'cache': self.cache,
            'remaining_files': self.files
        }
        
        try:
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved {dataset_type} cache checkpoint with {len(self.cache)} sessions")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
    
    def _fill_cache(self):
        """Fill the cache with preprocessed sessions."""
        logger.info(f"Processing sessions for {'validation' if self.validation else 'training'} dataset")
        logger.info(f"Target cache size: {self.cache_size}, Files to process: {len(self.files)}")
        
        pbar = tqdm(total=self.cache_size, desc="Processing sessions")
        checkpoint_frequency = 100  # Save checkpoint every 100 sessions
        
        try:
            # Process files in batches for efficiency
            batch_utterances = []
            batch_indices = []
            max_batch_size = 32  # Process 32 utterances at once
            
            while len(self.cache) < self.cache_size and self.files:
                file_path = self.files.pop(0)
                logger.debug(f"Processing file: {file_path}")
                
                try:
                    with open(file_path, 'r') as f:
                        sessions = json.load(f)
                        for session in sessions:
                            if len(self.cache) >= self.cache_size:
                                break
                            
                            # Collect utterances for batch processing
                            sequence = []
                            for utterance in session['dialog']:
                                batch_utterances.append(utterance)
                                batch_indices.append(len(sequence))
                                sequence.append(None)  # Placeholder
                                
                                # Process batch when it reaches max size
                                if len(batch_utterances) >= max_batch_size:
                                    embeddings = self._process_utterance_batch(batch_utterances)
                                    self._update_sequences(embeddings, batch_indices, sequence)
                                    batch_utterances = []
                                    batch_indices = []
                            
                            # Process remaining utterances in the session
                            if batch_utterances:
                                embeddings = self._process_utterance_batch(batch_utterances)
                                self._update_sequences(embeddings, batch_indices, sequence)
                                batch_utterances = []
                                batch_indices = []
                            
                            self.cache.append(sequence)
                            pbar.update(1)
                            
                            # Save checkpoint periodically
                            if len(self.cache) % checkpoint_frequency == 0:
                                self._save_checkpoint()
                                
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
                    continue
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user, saving checkpoint...")
            self._save_checkpoint()
            raise
        
        finally:
            pbar.close()
            # Save final checkpoint
            self._save_checkpoint()
            logger.info(f"Cached {len(self.cache)} sessions")
    
    def _process_utterance_batch(self, utterances):
        """Process a batch of utterances through BERT."""
        inputs = self.tokenizer(
            utterances,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :]
    
    def _update_sequences(self, embeddings, indices, sequence):
        """Update sequence with computed embeddings."""
        for embedding, idx in zip(embeddings, indices):
            sequence[idx] = embedding
    
    def __len__(self):
        return len(self.cache)
    
    def __getitem__(self, idx):
        if idx not in self.cache_indices and len(self.cache) < self.cache_size:
            self._fill_cache()
        return self.cache[idx]

def collate_sessions(batch: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function for batching sessions.
    
    Args:
        batch: List of sequences, where each sequence is a list of utterance embeddings
        
    Returns:
        Tuple of (padded_sequences, sequence_lengths)
    """
    # Get sequence lengths
    lengths = [len(seq) for seq in batch]
    max_len = max(lengths)
    
    # Pad sequences
    padded_seqs = []
    for seq in batch:
        # Pad with zero embeddings if sequence is shorter than max_len
        if len(seq) < max_len:
            padding = [torch.zeros_like(seq[0]) for _ in range(max_len - len(seq))]
            seq.extend(padding)
        padded_seqs.append(torch.stack(seq))
    
    # Stack into batch
    padded_batch = torch.stack(padded_seqs)
    lengths_tensor = torch.tensor(lengths, device=padded_batch.device)
    
    return padded_batch, lengths_tensor

def train_ees(
    train_dataset: TherapySessionDataset,
    val_dataset: TherapySessionDataset = None,
    input_dim: int = 768,
    latent_dim: int = 32,
    hidden_dim: int = 128,
    batch_size: int = 16,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir: str = "checkpoints"
):
    """Train the Emotional Embedding Space model using batched data."""
    
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
    
    # Create data loaders with batching
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sessions
    )
    
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_sessions
        )
    
    logger.info(f"Starting training with {num_epochs} epochs")
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        ees.train()
        memory_system.train()
        
        total_loss = 0
        num_batches = 0
        
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (batch_sequences, sequence_lengths) in enumerate(train_iter):
            optimizer.zero_grad()
            
            batch_loss = 0
            
            # Process each sequence in the batch
            for i in range(batch_sequences.size(0)):
                seq_len = sequence_lengths[i]
                sequence = batch_sequences[i, :seq_len]  # Only use non-padded parts
                
                # Reset state for new sequence
                ees.previous_state = None
                memory_context = torch.zeros(1, input_dim).to(device)
                
                # Process each utterance
                for utterance_embedding in sequence:
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
            batch_loss = batch_loss / (batch_sequences.size(0) * sequence_lengths.float().mean())
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}: Loss = {batch_loss.item():.4f}")
            
            train_iter.set_postfix({'loss': batch_loss.item()})
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Validation
        if val_dataset:
            val_loss = validate_ees(ees, memory_system, val_loader, device)
            logger.info(f"Validation Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'ees_state_dict': ees.state_dict(),
                    'memory_state_dict': memory_system.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss
                }, f"{checkpoint_dir}/best_model.pt")
                logger.info(f"Saved checkpoint for epoch {epoch+1}")
    
    return ees, memory_system

def validate_ees(ees, memory_system, val_loader, device):
    """Validate the EES model using batched validation data."""
    ees.eval()
    memory_system.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (batch_sequences, sequence_lengths) in enumerate(val_loader):
            batch_loss = 0
            
            # Process each sequence in the batch
            for i in range(batch_sequences.size(0)):
                seq_len = sequence_lengths[i]
                sequence = batch_sequences[i, :seq_len]
                
                ees.previous_state = None
                memory_context = torch.zeros(1, ees.input_dim).to(device)
                
                for utterance_embedding in sequence:
                    bert_context = utterance_embedding
                    loss = ees.compute_loss(
                        utterance_embedding,
                        bert_context,
                        memory_context
                    )
                    batch_loss += loss
                    memory_context = memory_system(
                        utterance_embedding,
                        memory_context
                    )
            
            batch_loss = batch_loss / (batch_sequences.size(0) * sequence_lengths.float().mean())
            total_loss += batch_loss.item()
            num_batches += 1
        
        return total_loss / num_batches

if __name__ == "__main__":
    # Initialize BERT model and tokenizer
    logger.info("Initializing BERT model and tokenizer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model = AutoModel.from_pretrained("bert-base-uncased").to(device)
    logger.info("BERT initialization complete")
    
    # Create datasets with batching
    logger.info("Creating training and validation datasets...")
    max_sessions = 10000  # Limit to 10k sessions for initial training
    batch_size = 16
    logger.info(f"Using batch size: {batch_size}, max sessions: {max_sessions}")
    
    logger.info("Initializing training dataset...")
    train_dataset = TherapySessionDataset(
        data_dir="data/generated_sessions",
        tokenizer=tokenizer,
        model=bert_model,
        device=device,
        max_sessions=max_sessions,
        validation=False,
        batch_size=batch_size,
        checkpoint_dir="data/checkpoints"
    )
    logger.info(f"Training dataset initialized with {len(train_dataset)} sessions")
    
    logger.info("Initializing validation dataset...")
    val_dataset = TherapySessionDataset(
        data_dir="data/generated_sessions",
        tokenizer=tokenizer,
        model=bert_model,
        device=device,
        max_sessions=max_sessions,
        validation=True,
        batch_size=batch_size,
        checkpoint_dir="data/checkpoints"
    )
    logger.info(f"Validation dataset initialized with {len(val_dataset)} sessions")
    
    # Create checkpoints directory
    Path("checkpoints").mkdir(exist_ok=True)
    logger.info("Created checkpoints directory")
    
    # Train model
    logger.info("Starting training...")
    ees, memory_system = train_ees(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        batch_size=batch_size,
        checkpoint_dir="checkpoints"
    )
    
    logger.info("Training complete!")
