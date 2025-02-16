"""Training script for the Emotional Embedding Space."""

import numpy as np
import h5py
import json
import random
import glob
import logging
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Iterator
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.emotional_embedding.ees import EmotionalEmbeddingSpace
from src.memory.memory_system import TransformerMemorySystem

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
        cache_size: int = 1000,
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
        self._pool = None  # Will be initialized when needed
        
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
    
    def cleanup(self):
        """Clean up resources used by the dataset."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear the cache to free memory
        self.cache.clear()
        self.cache_indices.clear()
    
    def _load_or_create_cache(self):
        """Load cache from checkpoint if exists, otherwise create new cache."""
        dataset_type = "validation" if self.validation else "training"
        checkpoint_path = Path(self.checkpoint_dir) / f"dataset_cache_{dataset_type}.h5"
        
        if checkpoint_path.exists():
            logger.info(f"Found existing checkpoint at {checkpoint_path}")
            try:
                with h5py.File(checkpoint_path, 'r') as f:
                    num_sessions = f['num_sessions'][()]
                    logger.info(f"Checkpoint contains {num_sessions} sessions")
                    
                    # Load sessions into cache
                    sessions_to_load = min(num_sessions, self.cache_size)
                    logger.info(f"Loading {sessions_to_load} sessions into cache")
                    
                    for i in range(sessions_to_load):
                        session_group = f[f'session_{i}']
                        sequence = []
                        num_utterances = session_group['num_utterances'][()]
                        
                        for j in range(num_utterances):
                            embedding = torch.from_numpy(session_group[f'utterance_{j}'][()]).float()
                            sequence.append(embedding)
                        
                        self.cache.append(sequence)
                        if (i + 1) % 100 == 0:
                            logger.info(f"Loaded {i + 1} sessions")
                    
                    # Load remaining files
                    self.files = json.loads(f['remaining_files'][()])
                    logger.info(f"Loaded {len(self.files)} remaining files to process")
                
                logger.info(f"Successfully loaded {len(self.cache)} cached sessions")
                return
                
            except Exception as e:
                logger.error(f"Error loading checkpoint: {str(e)}")
                logger.error("Falling back to creating new cache")
        
        logger.info(f"No valid checkpoint found at {checkpoint_path}, creating new cache")
        self._fill_cache()
    
    def _save_checkpoint(self):
        """Save current cache state to checkpoint."""
        dataset_type = "validation" if self.validation else "training"
        checkpoint_path = Path(self.checkpoint_dir) / f"dataset_cache_{dataset_type}.h5"
        
        try:
            with h5py.File(checkpoint_path, 'w') as f:
                f.create_dataset('num_sessions', data=len(self.cache))
                
                # Save sessions
                for i, sequence in enumerate(self.cache):
                    session_group = f.create_group(f'session_{i}')
                    session_group.create_dataset('num_utterances', data=len(sequence))
                    
                    for j, embedding in enumerate(sequence):
                        session_group.create_dataset(
                            f'utterance_{j}',
                            data=embedding.cpu().numpy()
                        )
                
                # Save remaining files
                f.create_dataset(
                    'remaining_files',
                    data=json.dumps(self.files)
                )
            
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
        """Get a sequence from the dataset.
        
        Args:
            idx: Index of the sequence to get
            
        Returns:
            List of utterance embeddings for the sequence
        """
        sequence = self.cache[idx]
        return [emb.clone().detach().float() for emb in sequence]

def collate_sessions(batch: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function for batching sessions.
    
    Args:
        batch: List of sequences, where each sequence is a list of utterance embeddings
        
    Returns:
        Tuple of (padded_sequences, sequence_lengths)
    """
    # Get lengths of each sequence
    lengths = torch.tensor([len(seq) for seq in batch])
    max_len = max(lengths)
    
    # Get embedding dimension from first utterance of first sequence
    emb_dim = batch[0][0].size(-1)
    
    # Create padded tensor
    padded = torch.zeros(len(batch), max_len, emb_dim)
    
    # Fill padded tensor with sequences
    for i, sequence in enumerate(batch):
        seq_len = lengths[i]
        sequence_tensor = torch.stack([utt.clone().detach() for utt in sequence])
        padded[i, :seq_len] = sequence_tensor
    
    return padded, lengths

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
    optimizer = torch.optim.Adam(
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
        total_emotional_coherence = 0  # Measure how well emotions flow
        total_contextual_alignment = 0  # Measure alignment with conversation context
        total_batches = 0
        
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (batch_sequences, sequence_lengths) in enumerate(train_iter):
            optimizer.zero_grad()
            
            batch_loss = 0
            batch_emotional_coherence = 0
            batch_contextual_alignment = 0
            
            # Process each sequence in the batch
            for i in range(batch_sequences.size(0)):
                seq_len = sequence_lengths[i]
                sequence = batch_sequences[i, :seq_len].to(device)  # Move sequence to device
                
                ees.previous_state = None
                memory_context = torch.zeros(1, input_dim, device=device)  # Use device parameter
                
                # Process each utterance
                for utterance_embedding in sequence:
                    utterance_embedding = utterance_embedding.to(device)  # Ensure on device
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
                    
                    # Compute emotional coherence and contextual alignment
                    if ees.previous_state is not None:
                        emotional_coherence = ees.emotional_transition_loss(
                            ees.previous_state,
                            utterance_embedding
                        )
                        batch_emotional_coherence += emotional_coherence.item()
                        
                        context_alignment = ees.contextual_alignment_loss(
                            utterance_embedding,
                            [ees.previous_state, memory_context]
                        )
                        batch_contextual_alignment += context_alignment.item()
                    
                    ees.previous_state = utterance_embedding
            
            # Average loss over batch
            batch_loss = batch_loss / (batch_sequences.size(0) * sequence_lengths.float().mean())
            batch_emotional_coherence = batch_emotional_coherence / (batch_sequences.size(0) * sequence_lengths.float().mean())
            batch_contextual_alignment = batch_contextual_alignment / (batch_sequences.size(0) * sequence_lengths.float().mean())
            
            batch_loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += batch_loss.item()
            total_emotional_coherence += batch_emotional_coherence
            total_contextual_alignment += batch_contextual_alignment
            total_batches += 1
            
            # Update progress bar
            train_iter.set_postfix({
                'Loss': f'{batch_loss.item():.4f}',
                'E-Coherence': f'{batch_emotional_coherence:.4f}',
                'C-Alignment': f'{batch_contextual_alignment:.4f}'
            })
        
        # Calculate epoch averages
        avg_loss = total_loss / total_batches if total_batches > 0 else float('inf')
        avg_coherence = total_emotional_coherence / total_batches if total_batches > 0 else float('inf')
        avg_alignment = total_contextual_alignment / total_batches if total_batches > 0 else float('inf')
        
        logger.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}, Emotional Coherence: {avg_coherence:.4f}, Contextual Alignment: {avg_alignment:.4f}")
        
        # Validation
        if val_dataset:
            val_loss, val_emotional_coherence, val_contextual_alignment = validate_ees(ees, memory_system, val_loader, device)
            logger.info(f"Validation Loss: {val_loss:.4f}, Emotional Coherence: {val_emotional_coherence:.4f}, Contextual Alignment: {val_contextual_alignment:.4f}")
            
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
    ees.eval()
    memory_system.eval()
    total_loss = 0
    total_emotional_coherence = 0
    total_contextual_alignment = 0
    total_batches = 0
    
    with torch.no_grad():
        for batch_idx, (batch_sequences, sequence_lengths) in enumerate(val_loader):
            batch_loss = 0
            batch_emotional_coherence = 0
            batch_contextual_alignment = 0
            
            # Process each sequence in the batch
            for i in range(batch_sequences.size(0)):
                seq_len = sequence_lengths[i]
                sequence = batch_sequences[i, :seq_len]
                
                ees.previous_state = None
                memory_context = torch.zeros(1, ees.input_dim).to(device)
                
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
                    
                    # Compute emotional coherence and contextual alignment
                    if ees.previous_state is not None:
                        emotional_coherence = ees.emotional_transition_loss(
                            ees.previous_state,
                            utterance_embedding
                        )
                        batch_emotional_coherence += emotional_coherence.item()
                        
                        context_alignment = ees.contextual_alignment_loss(
                            utterance_embedding,
                            [ees.previous_state, memory_context]
                        )
                        batch_contextual_alignment += context_alignment.item()
                    
                    ees.previous_state = utterance_embedding
            
            # Average loss over batch
            batch_loss = batch_loss / (batch_sequences.size(0) * sequence_lengths.float().mean())
            batch_emotional_coherence = batch_emotional_coherence / (batch_sequences.size(0) * sequence_lengths.float().mean())
            batch_contextual_alignment = batch_contextual_alignment / (batch_sequences.size(0) * sequence_lengths.float().mean())
            
            total_loss += batch_loss.item()
            total_emotional_coherence += batch_emotional_coherence
            total_contextual_alignment += batch_contextual_alignment
            total_batches += 1
    
    avg_loss = total_loss / total_batches if total_batches > 0 else float('inf')
    avg_coherence = total_emotional_coherence / total_batches if total_batches > 0 else float('inf')
    avg_alignment = total_contextual_alignment / total_batches if total_batches > 0 else float('inf')
    
    return avg_loss, avg_coherence, avg_alignment

if __name__ == "__main__":
    try:
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        
        # Set up device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Training parameters
        batch_size = 16
        max_sessions = 10000
        
        # Initialize datasets with proper resource management
        train_dataset = None
        val_dataset = None
        
        try:
            train_dataset = TherapySessionDataset(
                data_dir="data/generated_sessions",
                tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased'),
                model=AutoModel.from_pretrained('bert-base-uncased'),
                device=device,
                max_sessions=max_sessions,
                validation=False,
                batch_size=batch_size,
                checkpoint_dir="data/checkpoints"
            )
            
            val_dataset = TherapySessionDataset(
                data_dir="data/generated_sessions",
                tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased'),
                model=AutoModel.from_pretrained('bert-base-uncased'),
                device=device,
                max_sessions=max_sessions // 5,  # Smaller validation set
                validation=True,
                batch_size=batch_size,
                checkpoint_dir="data/checkpoints"
            )
            
            # Train the model
            train_ees(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                device=device,
                batch_size=batch_size,
                num_epochs=10,
                learning_rate=1e-4
            )
            
        finally:
            # Cleanup resources
            if train_dataset is not None:
                train_dataset.cleanup()
            if val_dataset is not None:
                val_dataset.cleanup()
            
            # Force cleanup of any remaining multiprocessing resources
            import multiprocessing
            if hasattr(multiprocessing, '_cleanup'):
                multiprocessing._cleanup()
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
