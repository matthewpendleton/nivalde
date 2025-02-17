"""
Temporal Parser for handling flexible natural language time expressions.

Handles:
1. Relative time expressions ("last week", "few months ago")
2. Ordinal relationships ("before my wedding", "after I moved")
3. Fuzzy boundaries ("around Christmas", "during summer")
4. Nested temporal relationships ("before the time when...")
5. Uncertainty calibration based on context
"""

import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import torch
import numpy as np
from datetime import datetime, timedelta
import re

@dataclass
class TemporalReference:
    """A temporal reference with uncertainty."""
    mean_time: float  # Estimated mean time in seconds since epoch
    uncertainty: float  # Standard deviation in days
    confidence: float  # Overall confidence in estimate (0-1)
    ordinal_relations: List['OrdinalLink']  # Links to other temporal references
    context_weight: float  # How much context influences this reference (0-1)

@dataclass
class OrdinalLink:
    """Represents an ordinal relationship between temporal references."""
    target_ref: 'TemporalReference'
    relation_type: str  # 'before', 'after', 'during', 'around'
    confidence: float
    temporal_distance: Optional[float]  # Estimated time difference in days

class TemporalContext:
    """Maintains context for temporal reasoning."""
    
    def __init__(self):
        self.anchor_points: Dict[str, TemporalReference] = {}
        self.recent_references: List[TemporalReference] = []
        
    def add_anchor(self, name: str, reference: TemporalReference):
        self.anchor_points[name] = reference
        
    def get_relevant_anchors(self, expr: str) -> List[Tuple[str, TemporalReference]]:
        """Find contextually relevant anchor points for a temporal expression."""
        return [(name, ref) for name, ref in self.anchor_points.items()
                if self._is_relevant(expr, name)]
    
    def _is_relevant(self, expr: str, anchor_name: str) -> bool:
        """Determine if an anchor point is relevant to an expression."""
        # Simple keyword matching for now
        return any(word in expr.lower() for word in anchor_name.lower().split())

class TemporalParser:
    """Parser for natural language temporal expressions."""
    
    # Basic temporal patterns with uncertainty estimates
    PATTERNS = {
        r"yesterday|last\s+night": (1, 0.2),  # 1 day ago, low uncertainty
        r"last\s+week": (7, 2.0),
        r"last\s+month": (30, 7.0),
        r"few\s+months\s+ago": (90, 30.0),
        r"last\s+year": (365, 60.0),
        r"years?\s+ago": (730, 180.0),
        r"childhood|when\s+I\s+was\s+young": (7300, 1825.0),  # ~20 years, 5 year uncertainty
        r"long\s+ago|ages?\s+ago": (3650, 730.0)
    }
    
    # Seasonal patterns
    SEASONAL_PATTERNS = {
        r"summer": ((6, 8), 30.0),  # (month range, uncertainty in days)
        r"winter": ((12, 2), 30.0),
        r"spring": ((3, 5), 30.0),
        r"fall|autumn": ((9, 11), 30.0)
    }
    
    # Ordinal relation patterns
    ORDINAL_PATTERNS = {
        r"before": "before",
        r"after|following": "after",
        r"during|while": "during",
        r"around|about": "around"
    }
    
    # Life stage patterns with broad uncertainty
    LIFE_STAGES = {
        r"as\s+a\s+child|in\s+childhood": (7300, 1825.0, 0.7),  # ~20 years ago, 5 year uncertainty
        r"teenager|teens|adolescence": (5475, 1095.0, 0.7),  # ~15 years ago, 3 year uncertainty
        r"college|university": (3650, 730.0, 0.8),  # ~10 years ago, 2 year uncertainty
        r"early\s+twenties": (2920, 730.0, 0.8),
        r"growing\s+up": (5475, 2190.0, 0.6),  # Very broad period
    }
    
    # Emotional anchoring patterns
    EMOTIONAL_ANCHORS = {
        r"happiest|best\s+time": "positive_peak",
        r"worst|darkest\s+time": "negative_peak",
        r"turning\s+point|life\s+changed": "transition",
        r"breakthrough|revelation": "insight",
    }
    
    # Frequency and repetition patterns
    FREQUENCY_PATTERNS = {
        r"every\s+day": (1, 0.2, "regular"),
        r"weekly|every\s+week": (7, 1.0, "regular"),
        r"monthly": (30, 5.0, "regular"),
        r"occasionally|sometimes": (14, 7.0, "irregular"),
        r"rarely|seldom": (90, 30.0, "irregular"),
    }
    
    # Duration patterns
    DURATION_PATTERNS = {
        r"for\s+a\s+while": (30, 15.0),
        r"brief|quickly": (2, 1.0),
        r"long\s+time": (180, 90.0),
        r"forever|always": (365, 180.0),
    }
    
    # Relational intensity patterns
    INTENSITY_RELATIONS = {
        r"just|recently": 0.9,  # High temporal precision
        r"about|around": 0.7,   # Medium precision
        r"sometime": 0.5,       # Low precision
        r"maybe|possibly": 0.3,  # Very low precision
    }
    
    # Cultural/Personal Event Patterns
    CULTURAL_EVENTS = {
        r"christmas|holiday": ((12, 25), 5.0),
        r"new\s+year": ((1, 1), 2.0),
        r"birthday": "personal_annual",
        r"graduation": "life_milestone",
        r"wedding": "life_milestone",
    }
    
    # Temporal Clustering Patterns
    CLUSTER_PATTERNS = {
        r"that\s+same\s+time": "concurrent",
        r"right\s+after": "immediate_follow",
        r"series\s+of": "sequence",
        r"one\s+after\s+another": "sequence",
    }
    
    def __init__(self):
        self.context = TemporalContext()
        self.personal_anchors = {}  # Store personal recurring events
        self.emotional_peaks = {}   # Store emotional peak experiences
        
    def parse_expression(
        self,
        expr: str,
        base_confidence: float = 1.0,
        emotional_valence: Optional[float] = None
    ) -> TemporalReference:
        """Parse temporal expression with enhanced pattern matching."""
        
        # 1. Check for direct pattern matches
        for pattern, (days_ago, uncertainty) in self.PATTERNS.items():
            if re.search(pattern, expr, re.I):
                return self._create_reference(days_ago, uncertainty, base_confidence)
        
        # 2. Check for seasonal references
        for pattern, ((start_month, end_month), uncertainty) in self.SEASONAL_PATTERNS.items():
            if re.search(pattern, expr, re.I):
                return self._create_seasonal_reference(start_month, end_month, uncertainty)
        
        # 3. Check for ordinal relationships
        ordinal_ref = self._parse_ordinal_relation(expr)
        if ordinal_ref:
            return ordinal_ref
        
        # 4. Check life stage patterns
        for pattern, (days_ago, uncertainty, conf) in self.LIFE_STAGES.items():
            if re.search(pattern, expr, re.I):
                return self._create_reference(
                    days_ago,
                    uncertainty,
                    base_confidence * conf
                )
        
        # 5. Check emotional anchoring
        for pattern, anchor_type in self.EMOTIONAL_ANCHORS.items():
            if re.search(pattern, expr, re.I):
                return self._handle_emotional_anchor(
                    anchor_type,
                    emotional_valence,
                    base_confidence
                )
        
        # 6. Check frequency patterns
        freq_ref = self._parse_frequency(expr, base_confidence)
        if freq_ref:
            return freq_ref
            
        # 7. Handle cultural/personal events
        event_ref = self._parse_cultural_event(expr, base_confidence)
        if event_ref:
            return event_ref
            
        # 8. Check temporal clusters
        cluster_ref = self._parse_cluster(expr, base_confidence)
        if cluster_ref:
            return cluster_ref
            
        # 9. Use context for unknown patterns
        return self._infer_from_context(expr, base_confidence)
    
    def _create_reference(
        self,
        days_ago: float,
        uncertainty: float,
        confidence: float
    ) -> TemporalReference:
        """Create a temporal reference from days ago."""
        now = time.time()
        mean_time = now - (days_ago * 24 * 3600)
        return TemporalReference(
            mean_time=mean_time,
            uncertainty=uncertainty,
            confidence=confidence,
            ordinal_relations=[],
            context_weight=0.3
        )
    
    def _create_seasonal_reference(
        self,
        start_month: int,
        end_month: int,
        uncertainty: float
    ) -> TemporalReference:
        """Create a reference for seasonal expressions."""
        now = datetime.now()
        year = now.year
        
        # Handle winter crossing year boundary
        if start_month > end_month and now.month < 6:
            year -= 1
            
        start_date = datetime(year, start_month, 1)
        mean_time = start_date.timestamp() + ((end_month - start_month) * 30 * 24 * 3600 / 2)
        
        return TemporalReference(
            mean_time=mean_time,
            uncertainty=uncertainty,
            confidence=0.7,  # Lower confidence for seasonal references
            ordinal_relations=[],
            context_weight=0.5
        )
    
    def _parse_ordinal_relation(self, expr: str) -> Optional[TemporalReference]:
        """Parse expressions with ordinal relationships."""
        for pattern, relation in self.ORDINAL_PATTERNS.items():
            match = re.search(f"({pattern})\\s+(.+)", expr, re.I)
            if match:
                relation_type = relation
                target_expr = match.group(2)
                
                # Find relevant anchor points
                anchors = self.context.get_relevant_anchors(target_expr)
                if not anchors:
                    # If no anchors, parse the target expression itself
                    target_ref = self.parse_expression(target_expr, 0.8)
                else:
                    # Use the most relevant anchor
                    target_ref = anchors[0][1]
                
                # Create the ordinal relationship
                link = OrdinalLink(
                    target_ref=target_ref,
                    relation_type=relation_type,
                    confidence=0.8,
                    temporal_distance=self._estimate_temporal_distance(relation_type)
                )
                
                # Adjust mean time based on relation
                mean_time = self._adjust_time_for_relation(
                    target_ref.mean_time,
                    link
                )
                
                return TemporalReference(
                    mean_time=mean_time,
                    uncertainty=target_ref.uncertainty * 1.5,  # Increase uncertainty for ordinal refs
                    confidence=target_ref.confidence * 0.9,
                    ordinal_relations=[link],
                    context_weight=0.6
                )
        
        return None
    
    def _estimate_temporal_distance(self, relation_type: str) -> Optional[float]:
        """Estimate typical temporal distance for a relation type in days."""
        distances = {
            "before": 30.0,  # Default to one month
            "after": 30.0,
            "during": 0.0,
            "around": 7.0
        }
        return distances.get(relation_type)
    
    def _adjust_time_for_relation(
        self,
        base_time: float,
        link: OrdinalLink
    ) -> float:
        """Adjust time based on ordinal relationship."""
        if not link.temporal_distance:
            return base_time
            
        adjustment = link.temporal_distance * 24 * 3600  # Convert days to seconds
        
        if link.relation_type == "before":
            return base_time - adjustment
        elif link.relation_type == "after":
            return base_time + adjustment
        else:
            return base_time  # No adjustment for during/around
    
    def _handle_emotional_anchor(
        self,
        anchor_type: str,
        valence: Optional[float],
        confidence: float
    ) -> TemporalReference:
        """Handle emotionally anchored temporal references."""
        if anchor_type in self.emotional_peaks:
            peak = self.emotional_peaks[anchor_type]
            # Adjust uncertainty based on emotional intensity
            uncertainty = 30.0 * (1.0 - abs(valence or 0.5))
            return TemporalReference(
                mean_time=peak.mean_time,
                uncertainty=uncertainty,
                confidence=confidence * 0.9,
                ordinal_relations=[],
                context_weight=0.8
            )
        return self._create_reference(90, 45.0, confidence * 0.6)
        
    def _parse_frequency(
        self,
        expr: str,
        confidence: float
    ) -> Optional[TemporalReference]:
        """Parse frequency-based temporal expressions."""
        for pattern, (period, uncertainty, freq_type) in self.FREQUENCY_PATTERNS.items():
            if re.search(pattern, expr, re.I):
                # For regular patterns, create multiple reference points
                if freq_type == "regular":
                    references = []
                    for i in range(5):  # Last 5 occurrences
                        time_ago = period * (i + 1)
                        ref = self._create_reference(time_ago, uncertainty, confidence)
                        references.append(ref)
                    # Return most recent with links to others
                    return TemporalReference(
                        mean_time=references[0].mean_time,
                        uncertainty=uncertainty,
                        confidence=confidence,
                        ordinal_relations=[
                            OrdinalLink(ref, "before", 0.9, period)
                            for ref in references[1:]
                        ],
                        context_weight=0.7
                    )
                else:
                    # For irregular patterns, increase uncertainty
                    return self._create_reference(period, uncertainty * 2, confidence * 0.8)
        return None
        
    def _parse_cultural_event(
        self,
        expr: str,
        confidence: float
    ) -> Optional[TemporalReference]:
        """Parse cultural and personal event references."""
        for pattern, event_type in self.CULTURAL_EVENTS.items():
            if re.search(pattern, expr, re.I):
                if isinstance(event_type, tuple):
                    # Fixed calendar events
                    month, day = event_type[0]
                    uncertainty = event_type[1]
                    return self._create_calendar_reference(month, day, uncertainty, confidence)
                elif event_type == "personal_annual":
                    # Recurring personal events
                    if pattern in self.personal_anchors:
                        return self._create_recurring_reference(
                            self.personal_anchors[pattern],
                            365,  # Annual cycle
                            confidence
                        )
                elif event_type == "life_milestone":
                    # One-time significant events
                    if pattern in self.personal_anchors:
                        return self.personal_anchors[pattern]
        return None
        
    def _parse_cluster(
        self,
        expr: str,
        confidence: float
    ) -> Optional[TemporalReference]:
        """Parse temporal clustering expressions."""
        for pattern, cluster_type in self.CLUSTER_PATTERNS.items():
            if re.search(pattern, expr, re.I):
                if cluster_type == "concurrent":
                    # Use most recent reference with tight uncertainty
                    if self.context.recent_references:
                        ref = self.context.recent_references[-1]
                        return TemporalReference(
                            mean_time=ref.mean_time,
                            uncertainty=ref.uncertainty * 0.5,  # Tighter uncertainty
                            confidence=confidence * 0.9,
                            ordinal_relations=[
                                OrdinalLink(ref, "during", 0.95, 0)
                            ],
                            context_weight=0.9
                        )
                elif cluster_type in ["sequence", "immediate_follow"]:
                    # Create sequence of references
                    if self.context.recent_references:
                        base_ref = self.context.recent_references[-1]
                        gap = 1.0 if cluster_type == "immediate_follow" else 7.0
                        return TemporalReference(
                            mean_time=base_ref.mean_time + (gap * 24 * 3600),
                            uncertainty=base_ref.uncertainty * 1.2,
                            confidence=confidence * 0.85,
                            ordinal_relations=[
                                OrdinalLink(base_ref, "after", 0.9, gap)
                            ],
                            context_weight=0.8
                        )
        return None
        
    def add_personal_anchor(
        self,
        name: str,
        reference: TemporalReference,
        recurring: bool = False
    ):
        """Add a personal temporal anchor point."""
        self.personal_anchors[name] = reference
        if recurring:
            self.context.add_anchor(f"recurring_{name}", reference)
            
    def add_emotional_peak(
        self,
        peak_type: str,
        reference: TemporalReference
    ):
        """Add an emotional peak experience."""
        self.emotional_peaks[peak_type] = reference
        
    def _infer_from_context(
        self,
        expr: str,
        base_confidence: float
    ) -> TemporalReference:
        """Infer temporal reference from context when no direct match."""
        # Default to medium-term past with high uncertainty
        return TemporalReference(
            mean_time=time.time() - (90 * 24 * 3600),  # 3 months ago
            uncertainty=90.0,  # High uncertainty (3 months)
            confidence=base_confidence * 0.5,  # Reduce confidence for inference
            ordinal_relations=[],
            context_weight=0.8  # High context weight for inferred times
        )
    
    def add_context(self, name: str, reference: TemporalReference):
        """Add a temporal anchor point to the context."""
        self.context.add_anchor(name, reference)
