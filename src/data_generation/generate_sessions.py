"""Generate synthetic therapy sessions for training."""

import json
import random
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple

from src.testing.adversarial_clients import (
    PsychologicalProfileGenerator,
    TherapySessionSimulator,
    QualityController
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class SessionDataGenerator:
    def __init__(
        self,
        output_dir: str = "/Users/matthewpendleton/nivalde/data/generated_sessions",
        num_clients: int = 100_000,
        sessions_per_client: int = 10,
        min_quality_score: float = 0.7,
        save_interval: int = 1000  # Save stats every N clients
    ):
        self.output_dir = Path(output_dir)
        self.num_clients = num_clients
        self.sessions_per_client = sessions_per_client
        self.min_quality_score = min_quality_score
        self.save_interval = save_interval
        
        # Initialize generators
        self.profile_gen = PsychologicalProfileGenerator()
        self.session_sim = TherapySessionSimulator()
        self.quality_ctrl = QualityController()
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_dataset(self):
        """Generate full dataset of therapy sessions."""
        logger.info(f"Generating dataset with {self.num_clients:,} clients, "
                   f"{self.sessions_per_client} sessions each")
        
        # Track overall statistics
        dataset_stats = {
            'total_clients': 0,
            'total_sessions': 0,
            'rejected_sessions': 0,
            'disorder_distribution': {},
            'quality_scores': [],
            'start_time': datetime.now().isoformat(),
            'last_update': datetime.now().isoformat()
        }
        
        # Generate clients with progress tracking
        progress_bar = tqdm(range(self.num_clients), desc="Generating clients")
        for client_id in progress_bar:
            # Generate client profile
            profile = self.profile_gen.generate_client_profile()
            
            # Update disorder stats
            for disorder in profile['disorders']:
                dataset_stats['disorder_distribution'][disorder] = \
                    dataset_stats['disorder_distribution'].get(disorder, 0) + 1
            
            # Generate sessions for this client
            client_sessions = []
            attempts = 0
            max_attempts = self.sessions_per_client * 2  # Allow some retries
            
            while len(client_sessions) < self.sessions_per_client and attempts < max_attempts:
                # Generate session
                dialog, outcome = self.session_sim.conduct_session(profile)
                
                # Evaluate quality
                quality_analysis = self.quality_ctrl.assess_session(dialog, outcome)
                quality_score = quality_analysis['authenticity_score'] / 5.0  # Normalize to 0-1
                
                if quality_score >= self.min_quality_score:
                    # Session passed quality check
                    session_data = {
                        'client_id': client_id,
                        'session_id': len(client_sessions),
                        'timestamp': datetime.now().isoformat(),
                        'profile': profile,
                        'dialog': dialog,
                        'outcome': outcome,
                        'quality_score': quality_score
                    }
                    
                    client_sessions.append(session_data)
                    dataset_stats['quality_scores'].append(quality_score)
                else:
                    dataset_stats['rejected_sessions'] += 1
                
                attempts += 1
            
            # Save client's sessions
            if client_sessions:
                client_file = self.output_dir / f"client_{client_id:06d}.json"
                with open(client_file, 'w') as f:
                    json.dump(client_sessions, f, indent=2, cls=DateTimeEncoder)
                
                dataset_stats['total_clients'] += 1
                dataset_stats['total_sessions'] += len(client_sessions)
            
            # Update progress bar with current stats
            if dataset_stats['total_clients'] > 0:
                avg_quality = sum(dataset_stats['quality_scores']) / len(dataset_stats['quality_scores'])
                progress_bar.set_postfix({
                    'clients': dataset_stats['total_clients'],
                    'sessions': dataset_stats['total_sessions'],
                    'avg_quality': f"{avg_quality:.3f}"
                })
            
            # Periodically save statistics
            if (client_id + 1) % self.save_interval == 0:
                dataset_stats['last_update'] = datetime.now().isoformat()
                stats_file = self.output_dir / "dataset_statistics.json"
                with open(stats_file, 'w') as f:
                    json.dump(dataset_stats, f, indent=2, cls=DateTimeEncoder)
        
        # Save final statistics
        dataset_stats['end_time'] = datetime.now().isoformat()
        stats_file = self.output_dir / "dataset_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(dataset_stats, f, indent=2, cls=DateTimeEncoder)
        
        # Calculate time taken
        start_time = datetime.fromisoformat(dataset_stats['start_time'])
        end_time = datetime.fromisoformat(dataset_stats['end_time'])
        duration = end_time - start_time
        
        # Log final summary
        logger.info(f"Dataset generation complete in {duration}:")
        logger.info(f"- Total clients: {dataset_stats['total_clients']:,}")
        logger.info(f"- Total sessions: {dataset_stats['total_sessions']:,}")
        logger.info(f"- Rejected sessions: {dataset_stats['rejected_sessions']:,}")
        logger.info(f"- Average quality score: {np.mean(dataset_stats['quality_scores']):.3f}")
        
        return dataset_stats

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Generate dataset
    generator = SessionDataGenerator(
        num_clients=100_000,        # 100k clients
        sessions_per_client=10,     # 10 sessions each
        min_quality_score=0.7,      # Minimum quality threshold
        save_interval=1000          # Save stats every 1000 clients
    )
    logging.info(f"Generating dataset with {generator.num_clients} clients, {generator.sessions_per_client} sessions each")
    stats = generator.generate_dataset()
