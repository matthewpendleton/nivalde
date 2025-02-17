"""Training script for the Emotional Embedding Space."""

import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

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
import gc
import multiprocessing as mp
import traceback
import psutil
import time
from datetime import datetime

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
        self._pool = None
        
        # Validate data directory
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory does not exist: {data_dir}")
        
        # Get list of all client files using os.listdir instead of glob
        data_path = Path(data_dir)
        try:
            all_files = os.listdir(data_path)
            self.files = sorted([
                str(data_path / f) for f in all_files
                if f.startswith('client_') and f.endswith('.json')
            ])
            logger.info(f"Found {len(all_files)} total files, {len(self.files)} matching client_*.json pattern")
        except Exception as e:
            raise ValueError(f"Error reading data directory {data_dir}: {str(e)}")
        
        if not self.files:
            raise ValueError(f"No client files found in {data_dir}. Files found: {all_files}")
            
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
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Ensure BERT model is in eval mode
        self.model.eval()
        
        # Try to load from checkpoint
        self._load_or_create_cache()
    
    def cleanup(self):
        """Clean up resources used by the dataset."""
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.join()
            except Exception as e:
                logger.warning(f"Error closing multiprocessing pool: {str(e)}")
            finally:
                self._pool = None
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Error clearing CUDA cache: {str(e)}")
        
        # Clear memory
        gc.collect()
    
    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        try:
            self.cleanup()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Error during cleanup in __del__: {str(e)}")
    
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
                    
                    if num_sessions > 0:
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
                        remaining_files = json.loads(f['remaining_files'][()])
                        if remaining_files:
                            self.files = remaining_files
                            logger.info(f"Loaded {len(self.files)} remaining files to process")
                        
                        logger.info(f"Successfully loaded {len(self.cache)} cached sessions")
                        return
                    else:
                        logger.info("Checkpoint is empty, creating new cache")
                
            except Exception as e:
                logger.error(f"Error loading checkpoint: {str(e)}")
                logger.error("Falling back to creating new cache")
        
        logger.info(f"Creating new cache for {dataset_type} dataset")
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
        
        start_time = time.time()
        pbar = tqdm(total=self.cache_size, desc="Processing sessions")
        checkpoint_frequency = 100  # Save checkpoint every 100 sessions
        
        try:
            # Process files in smaller batches for better memory management
            batch_utterances = []
            batch_indices = []
            max_batch_size = 16  # Reduced from 32 to 16 for better memory usage
            
            # Track memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
            
            while len(self.cache) < self.cache_size and self.files:
                file_path = self.files.pop(0)
                logger.debug(f"Processing file: {file_path}")
                
                try:
                    with open(file_path, 'r') as f:
                        sessions = json.load(f)
                        for session in sessions:
                            if len(self.cache) >= self.cache_size:
                                break
                            
                            try:
                                # Process utterances in smaller chunks
                                sequence = []
                                session_utterances = session['dialog']
                                
                                # Process utterances in chunks of max_batch_size
                                for i in range(0, len(session_utterances), max_batch_size):
                                    chunk = session_utterances[i:i + max_batch_size]
                                    chunk_indices = list(range(len(sequence), len(sequence) + len(chunk)))
                                    sequence.extend([None] * len(chunk))
                                    
                                    try:
                                        # Process chunk
                                        embeddings = self._process_utterance_batch(chunk)
                                        self._update_sequences(embeddings, chunk_indices, sequence)
                                    except Exception as chunk_error:
                                        logger.error(f"Error processing chunk: {str(chunk_error)}")
                                        continue
                                    finally:
                                        # Clear memory after each chunk
                                        if 'embeddings' in locals():
                                            del embeddings
                                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                                
                                # Add completed sequence to cache
                                if all(x is not None for x in sequence):
                                    self.cache.append([emb.cpu() for emb in sequence])
                                    pbar.update(1)
                                
                                    # Save checkpoint periodically
                                    if len(self.cache) % checkpoint_frequency == 0:
                                        current_memory = process.memory_info().rss / 1024 / 1024
                                        logger.info(f"Current memory usage: {current_memory:.2f} MB")
                                        self._save_checkpoint()
                                        logger.info(f"Saved checkpoint with {len(self.cache)} sessions")
                                else:
                                    logger.warning(f"Skipping incomplete sequence in {file_path}")
                                
                            except Exception as session_error:
                                logger.error(f"Error processing session: {str(session_error)}")
                                continue
                            
                except Exception as file_error:
                    logger.error(f"Error processing file {file_path}: {str(file_error)}")
                    continue
                finally:
                    # Clear memory after each file
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            logger.error(f"Error filling cache: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        finally:
            end_time = time.time()
            final_memory = process.memory_info().rss / 1024 / 1024
            logger.info(f"Final memory usage: {final_memory:.2f} MB")
            logger.info(f"Memory change: {final_memory - initial_memory:.2f} MB")
            logger.info(f"Total processing time: {(end_time - start_time) / 60:.2f} minutes")
            pbar.close()
            self._save_checkpoint()
    
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
    train_dataset,
    val_dataset,
    device,
    input_dim=768,
    hidden_dim=512,
    latent_dim=64,
    batch_size=16,
    num_epochs=20,
    learning_rate=1e-4
):
    """Train the Emotional Embedding Space model."""
    logger = logging.getLogger(__name__)
    
    # Ensure minimum batch size
    if batch_size < 4:
        logger.warning(f"Batch size {batch_size} is too small, setting to 4")
        batch_size = 4
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize models
    ees = EmotionalEmbeddingSpace(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim
    ).to(device)
    
    memory_system = TransformerMemorySystem(
        embedding_dim=input_dim
    ).to(device)
    
    # Initialize optimizer with gradient clipping
    optimizer = torch.optim.Adam(
        list(ees.parameters()) + list(memory_system.parameters()),
        lr=learning_rate
    )
    
    # Create data loaders with batching
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sessions,
        drop_last=True,
        num_workers=0  # Disable multiprocessing for now
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sessions,
        drop_last=True,
        num_workers=0
    )
    
    # Initialize training metrics
    best_val_loss = float('inf')
    start_time = time.time()
    process = psutil.Process()
    
    logger.info(f"Starting training with {num_epochs} epochs")
    logger.info(f"Initial memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    try:
        for epoch in range(num_epochs):
            epoch_start = time.time()
            train_loss = 0
            num_batches = 0
            
            # Training loop
            ees.train()
            memory_system.train()
            
            for batch_idx, (sequences, lengths) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                try:
                    sequences = sequences.to(device)
                    lengths = lengths.to(device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    loss = ees(sequences, lengths, memory_system)
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ees.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(memory_system.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    num_batches += 1
                    
                    # Clear memory
                    del sequences, lengths, loss
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                except Exception as batch_error:
                    logger.error(f"Error processing batch {batch_idx}: {str(batch_error)}")
                    continue
            
            avg_train_loss = train_loss / num_batches if num_batches > 0 else float('inf')
            
            # Validation
            val_loss = validate_ees(ees, memory_system, val_loader, device)
            
            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = checkpoint_dir / f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                torch.save({
                    'epoch': epoch,
                    'ees_state_dict': ees.state_dict(),
                    'memory_system_state_dict': memory_system.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)
                logger.info(f"Saved best model checkpoint to {checkpoint_path}")
            
            # Log epoch metrics
            epoch_time = time.time() - epoch_start
            current_memory = process.memory_info().rss / 1024 / 1024
            logger.info(f"Epoch {epoch+1}/{num_epochs}:")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            logger.info(f"  Time: {epoch_time:.2f}s")
            logger.info(f"  Memory: {current_memory:.2f} MB")
            
            # Clear memory after each epoch
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    
    finally:
        # Log final training metrics
        total_time = time.time() - start_time
        final_memory = process.memory_info().rss / 1024 / 1024
        logger.info(f"Training completed in {total_time/60:.2f} minutes")
        logger.info(f"Final memory usage: {final_memory:.2f} MB")
        
        # Cleanup
        train_dataset.cleanup()
        val_dataset.cleanup()

def validate_ees(ees, memory_system, val_loader, device):
    """Validate the EES model on validation data."""
    logger = logging.getLogger(__name__)
    ees.eval()
    memory_system.eval()
    total_loss = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch_sequences, sequence_lengths in val_loader:
            try:
                batch_size = batch_sequences.size(0)
                max_seq_len = batch_sequences.size(1)
                
                logger.debug(f"Processing validation batch: size={batch_size}, max_seq_len={max_seq_len}")
                
                # Move batch to device
                batch_sequences = batch_sequences.to(device)
                sequence_lengths = sequence_lengths.to(device)
                
                # Initialize batch states
                batch_memory = torch.zeros(batch_size, ees.input_dim, device=device)
                
                # Process each sequence in parallel
                for t in range(max_seq_len - 1):  # -1 because we need t+1 for next state
                    # Get current and next timestep embeddings
                    current_states = batch_sequences[:, t, :]
                    next_states = batch_sequences[:, t + 1, :]
                    
                    # Create mask for valid sequences at this timestep
                    current_mask = (t < sequence_lengths).float().unsqueeze(1)
                    next_mask = (t + 1 < sequence_lengths).float().unsqueeze(1)
                    
                    # Skip if no valid sequences at this timestep
                    if current_mask.sum() == 0 or next_mask.sum() == 0:
                        continue
                    
                    try:
                        # Compute loss only for valid sequences
                        loss = ees.compute_loss(
                            current_states,
                            next_states,
                            batch_memory
                        )
                        
                        # Apply mask and accumulate loss
                        valid_samples = current_mask.sum().item()
                        if valid_samples > 0:
                            masked_loss = (loss * current_mask).sum()
                            total_loss += masked_loss.item()
                            num_samples += valid_samples
                        
                        # Update memory
                        batch_memory = memory_system(current_states, batch_memory)
                        
                    except Exception as e:
                        logger.error(f"Error at timestep {t}: {str(e)}")
                        continue
                
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                continue
    
    # Compute average loss
    avg_loss = total_loss / num_samples if num_samples > 0 else 100.0
    
    # Log validation results
    logger.info(f"Validation complete - Samples: {num_samples}, Avg Loss: {avg_loss:.4f}")
    
    return avg_loss

if __name__ == "__main__":
    try:
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Set up the device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Initialize the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('bert-base-uncased').to(device)
        
        # Set up data directory
        data_dir = os.path.join(project_root, "data", "generated_sessions")
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory not found: {data_dir}")
        
        logger.info(f"Loading data from: {data_dir}")
        
        # Create training dataset with more data
        train_dataset = TherapySessionDataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            model=model,
            device=device,
            validation=False,
            cache_size=1000,  # Increased from 500
            batch_size=16,    # Increased from 8
            max_sessions=1000 # Increased from 500
        )
        
        # Clear memory before creating validation dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create validation dataset with proper size
        val_dataset = TherapySessionDataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            model=model,
            device=device,
            validation=True,
            cache_size=200,
            batch_size=16,
            max_sessions=200,  # Ensure enough validation data
            val_ratio=0.2      # Use 20% of data for validation
        )
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError(
                f"Dataset empty. Train size: {len(train_dataset)}, "
                f"Val size: {len(val_dataset)}. Ensure files match 'client_*.json'"
            )
        
        logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}")
        
        # Train the model with adjusted parameters
        train_ees(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            input_dim=768,
            hidden_dim=512,
            latent_dim=64,
            batch_size=16,
            num_epochs=20,
            learning_rate=1e-4
        )
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        # Cleanup
        if 'train_dataset' in locals():
            train_dataset.cleanup()
        if 'val_dataset' in locals():
            val_dataset.cleanup()
        
        # Final memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
