import random
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import numpy as np

class PsychologicalProfileGenerator:
    DISORDER_PREVALENCE = {
        # Primary Mental Health Disorders
        'generalized_anxiety': 0.18,
        'major_depression': 0.17,
        'social_anxiety': 0.12,
        'ptsd': 0.08,
        'bipolar': 0.04,
        'ocd': 0.03,
        'borderline_pd': 0.02,
        'dissociative_identity': 0.015,
        'paraphilic_disorders': 0.01,
        'schizophrenia': 0.007,
        
        # Life Stressors and Situational Challenges
        'existential_crisis': 0.25,
        'recent_bereavement': 0.15,
        'relationship_conflict': 0.32,
        'divorce_separation': 0.12,
        'parenting_stress': 0.18,
        'caregiver_burden': 0.09,
        'financial_crisis': 0.22,
        
        # Trauma History
        'childhood_abuse': 0.11,
        'domestic_violence': 0.08,
        'combat_trauma': 0.03,
        'medical_trauma': 0.06,
        
        # Personality Factors
        'perfectionism': 0.29,
        'chronic_shame': 0.17,
        'emotional_avoidance': 0.23
    }

    def generate_client_profile(self) -> Dict:
        """Generate a psychological profile with realistic comorbidities"""
        profile = {
            'disorders': self._select_disorders(),
            'demographics': self._generate_demographics(),
            'backstory': self._generate_backstory(),
            'weekly_events': self._generate_weekly_events(),
            'therapy_history': [],
            'current_state': 'pre_therapy'
        }
        return profile

    def _select_disorders(self) -> List[str]:
        disorders = []
        # Select primary disorders
        for disorder, prevalence in self.DISORDER_PREVALENCE.items():
            if random.random() < prevalence:
                # Group related disorders
                if disorder == 'paraphilic_disorders':
                    disorders.append(random.choice(['exhibitionism', 'voyeurism', 'fetishistic']))
                elif disorder == 'dissociative_identity':
                    disorders.extend(['depersonalization', 'derealization'])
                else:
                    disorders.append(disorder)
            
            # Limit to 3 primary disorders max
            if len([d for d in disorders if d in self.DISORDER_PREVALENCE.keys()]) >= 3:
                break

        # Add common life stressors (not counted in disorder limit)
        if random.random() < 0.78:  # % chance of at least one stressor
            stressors = [k for k in self.DISORDER_PREVALENCE.keys() 
                        if k in ['existential_crisis', 'recent_bereavement',
                                'relationship_conflict', 'divorce_separation',
                                'parenting_stress', 'caregiver_burden',
                                'financial_crisis']]
            disorders.extend(random.sample(stressors, k=random.randint(1,3)))

        return disorders if disorders else ['adjustment_disorder']

    def _generate_demographics(self) -> Dict:
        return {
            'age': random.randint(18, 65),
            'gender': random.choice(['male', 'female', 'non_binary']),
            'occupation': random.choice(['student', 'professional', 'unemployed', 'retired'])
        }

    def _generate_backstory(self) -> Dict:
        # Base template
        backstory = {
            'childhood': self._generate_childhood_history(),
            'adulthood': self._generate_adulthood_history(),
            'recent_life_events': self._generate_recent_events(),
            'ongoing_stressors': [],
            'cultural_factors': {
                'ethnicity': random.choice(['white', 'black', 'hispanic', 'asian', 'mixed']),
                'religion': random.choice(['christian', 'muslim', 'atheist', 'spiritual', 'other']),
                'immigrant_status': random.random() < 0.25
            },
            'family_relationships': self._generate_family_dynamics()
        }

        # Add trauma history if present
        if random.random() < 0.4:
            backstory['trauma_history'] = self._generate_trauma_history()

        # Add existential/spiritual elements
        if random.random() < 0.35:
            backstory['existential_concerns'] = random.choice([
                'meaning_of_life',
                'fear_of_death',
                'spiritual_crisis',
                'identity_questions'
            ])

        return backstory

    def _generate_childhood_history(self) -> Dict:
        return {
            'family_environment': random.choice(['stable', 'chaotic', 'abusive', 'neglectful']),
            'trauma_history': [self._generate_trauma_event() for _ in range(random.randint(0, 3))],
            'developmental_milestones': {
                'delayed': random.choice([True, False]),
                'areas': random.sample(['motor', 'speech', 'social'], random.randint(0, 3))
            }
        }

    def _generate_adulthood_history(self) -> Dict:
        return {
            'relationship_history': random.choice(['stable', 'volatile', 'isolated']),
            'career_path': random.choice(['linear', 'interrupted', 'multiple_changes']),
            'substance_use': random.choice(['never', 'experimental', 'regular', 'addiction']),
            'legal_issues': random.randint(0, 3)
        }

    def _generate_recent_events(self) -> List[Dict]:
        return [{
            'type': random.choice([
                'loss',
                'conflict',
                'achievement',
                'health_issue',
                'financial_change'
            ]),
            'severity': random.randint(1, 5),
            'timespan_days': random.randint(1, 180)
        } for _ in range(random.randint(1, 5))]

    def _generate_family_dynamics(self) -> Dict:
        return {
            'parent_status': random.choice(['both_alive', 'one_deceased', 'both_deceased']),
            'sibling_relationships': random.choice(['close', 'distant', 'conflictual']),
            'current_household': random.choice([
                'alone',
                'partner',
                'children',
                'extended_family',
                'roommates'
            ])
        }

    def _generate_trauma_history(self) -> Dict:
        return {
            'type': random.choice([
                'abuse',
                'accident',
                'disaster',
                'violence',
                'medical'
            ]),
            'age_at_incident': random.randint(0, 65),
            'chronicity': random.choice(['single', 'repeated']),
            'current_triggers': random.sample([
                'places',
                'people',
                'sounds',
                'dates',
                'sensory_cues'
            ], k=random.randint(0, 3))
        }

    def _generate_weekly_events(self) -> List[Dict]:
        events = []
        for _ in range(7):
            events.append({
                'date': datetime.now() - timedelta(days=random.randint(0,6)),
                'stress_level': random.randint(1,5),
                'social_interactions': random.randint(0,5),
                'trigger_encountered': random.choice([True, False])
            })
        return events

    def _generate_trauma_event(self):
        return {
            'type': random.choice(['medical', 'accident', 'violence', 'natural_disaster']),
            'age_at_event': random.randint(3, 65),
            'chronicity': random.choice(['single', 'repeated']),
            'disclosure_level': random.choice(['full', 'partial', 'denied'])
        }

class TherapySessionSimulator:
    def conduct_session(self, client_profile: Dict) -> Tuple[List[str], Dict]:
        """Simulate a therapy session between client and therapist AIs"""
        dialog = []
        session_outcome = {
            'therapist_approach': random.choice(['CBT', 'DBT', 'psychodynamic', 'humanistic']),
            'client_engagement': random.randint(1,5),
            'perceived_helpfulness': None
        }

        # Simulate session dialog
        for _ in range(5):  # 5 exchange turns
            therapist_line = self._generate_therapist_utterance(session_outcome['therapist_approach'])
            client_response = self._generate_client_response(therapist_line, client_profile)
            dialog.extend([f"Therapist: {therapist_line}", f"Client: {client_response}"])

        # Client evaluates session
        session_outcome['perceived_helpfulness'] = self._evaluate_session(dialog, client_profile)
        return dialog, session_outcome

    def _generate_therapist_utterance(self, modality: str) -> str:
        interventions = {
            'CBT': ["Let's examine the evidence for that thought",
                    "What alternative explanations might exist?"],
            'DBT': ["Let's practice some distress tolerance skills",
                    "How can we balance acceptance and change here?"],
            'psychodynamic': ["How does this relate to earlier life experiences?",
                            "What unconscious patterns might be at play?"],
            'humanistic': ["I sense you're feeling...",
                        "What does your ideal self look like in this situation?"]
        }
        return random.choice(interventions[modality])

    def _generate_client_response(self, therapist_line: str, profile: Dict) -> str:
        # Defense mechanisms based on personality and disorders
        defenses = self._identify_defense_mechanisms(profile)
        
        # Apply primary defense
        defense = random.choice(defenses)
        defended_response = self._apply_defense_mechanism(therapist_line, defense)

        # Then generate normal response
        base_response = self._generate_base_response(defended_response, profile)
        
        return base_response

    def _generate_base_response(self, therapist_line: str, profile: Dict) -> str:
        # Extract relevant backstory elements
        backstory = profile['backstory']
        response_pool = []

        # Add responses based on specific trauma history
        if 'trauma_history' in backstory:
            trauma = backstory['trauma_history']
            response_pool.extend([
                f"This reminds me of when I {trauma['type']}...",
                f"I've been having flashbacks to {trauma['age_at_incident']}",
                f"My {trauma['current_triggers']} have been bothering me"
            ])

        # Add responses based on family dynamics
        family = backstory['family_relationships']
        response_pool.extend([
            f"My {family['parent_status']} parents...",
            f"I feel {family['sibling_relationships']} with my siblings",
            f"Living with {family['current_household']} is stressful"
        ])

        # Add cultural considerations
        culture = backstory['cultural_factors']
        response_pool.extend([
            f"As a {culture['ethnicity']} person, I...",
            f"My {culture['religion']} beliefs...",
            "I don't feel understood culturally" if culture['immigrant_status'] else ""
        ])

        # Add existential concerns if present
        if 'existential_concerns' in backstory:
            response_pool.append(
                f"I keep wondering about {backstory['existential_concerns'].replace('_', ' ')}"
            )

        # Add modality-specific reactions
        modality = profile['therapy_history'][-1]['therapist_approach'] if profile['therapy_history'] else ''
        if modality == 'CBT':
            response_pool.append("I tried challenging that thought but...")
        elif modality == 'psychodynamic':
            response_pool.append("I don't see how my childhood relates...")

        # Select and personalize response
        if response_pool:
            base_response = random.choice(response_pool)
            return self._add_emotional_weight(base_response, profile)
        else:
            return "I'm not sure what to say..."

    def _add_emotional_weight(self, response: str, profile: Dict) -> str:
        # Add emotional valence based on disorders
        emotions = {
            'anxiety': ['worried', 'anxious', 'apprehensive'],
            'depression': ['hopeless', 'empty', 'numb'],
            'trauma': ['on edge', 'hypervigilant', 'triggered']
        }
        
        emotional_state = []
        for disorder, terms in emotions.items():
            if disorder in profile['disorders']:
                emotional_state.append(random.choice(terms))
        
        if emotional_state:
            return f"I'm feeling {random.choice(emotional_state)}... {response}"
        return response

    def _identify_defense_mechanisms(self, profile: Dict) -> List[str]:
        defenses = []
        
        # Personality-based defenses
        if 'perfectionism' in profile['disorders']:
            defenses.extend(['intellectualization', 'rationalization', 'moralization'])
        if 'emotional_avoidance' in profile['disorders']:
            defenses.extend(['repression', 'dissociation', 'isolation_of_affect'])
        if 'chronic_shame' in profile['disorders']:
            defenses.extend(['projection', 'turning_against_self', 'masochism'])
        
        # Trauma-related defenses
        trauma = profile['backstory'].get('trauma_history', {})
        if trauma.get('chronicity') == 'repeated':
            defenses.extend(['compartmentalization', 'dissociative_amnesia'])
        elif trauma.get('type') == 'medical':
            defenses.append('somatization')
            
        # Cultural defenses
        culture = profile['backstory']['cultural_factors']
        if culture['immigrant_status']:
            defenses.extend(['cultural_idealization', 'acculturation_conflict'])
        if culture['religion'] != 'atheist':
            defenses.extend(['spiritualization', 'religious_ritualization'])
        
        # Age-related defenses
        age = profile['demographics']['age']
        if age < 25:
            defenses.extend(['acting_out', 'fantasy'])
        elif age > 60:
            defenses.append('regression')
            
        return defenses if defenses else ['none']

    def _apply_defense_mechanism(self, therapist_line: str, defense: str) -> str:
        defense_transformations = {
            'intellectualization': lambda x: f"From a logical perspective, {x.lower()}",
            'rationalization': lambda x: "There's a good reason for that...",
            'repression': lambda x: "I don't remember...",
            'dissociation': lambda x: "I feel detached...",
            'denial': lambda x: "That's not really a problem...",
            'projection': lambda x: "Maybe you're feeling...",
            'splitting': lambda x: "You're either with me or against me...",
            'cultural_idealization': lambda x: "In my culture, we...",
            'spiritualization': lambda x: "It's all part of God's plan...",
            'moralization': lambda x: "It's really a moral issue...",
            'isolation_of_affect': lambda x: "(flatly) I suppose...",
            'turning_against_self': lambda x: "This is all my fault...",
            'masochism': lambda x: "I probably deserve this...",
            'compartmentalization': lambda x: "That's separate from...",
            'dissociative_amnesia': lambda x: "I don't recall...",
            'somatization': lambda x: "I've been having headaches...",
            'acculturation_conflict': lambda x: "I feel caught between cultures...",
            'religious_ritualization': lambda x: "Praying helps...",
            'acting_out': lambda x: "I don't want to talk about it!",
            'fantasy': lambda x: "In my ideal world...",
            'regression': lambda x: "I feel like a child...",
            'none': lambda x: x
        }
        return defense_transformations[defense](therapist_line)

    def _evaluate_session(self, dialog: List[str], profile: Dict) -> int:
        score = super()._evaluate_session(dialog, profile)
        
        # Cultural formulation assessment
        cultural_mentions = sum(1 for line in dialog if any(
            kw in line for kw in ['culture', 'ethnic', 'religion', 'immigrant']
        ))
        
        # Defense mechanism recognition
        defenses_recognized = sum(
            1 for line in dialog 
            if 'defense mechanism' in line.lower()
        )
        
        return min(5, score + cultural_mentions + defenses_recognized)

class QualityController:
    def assess_session(self, dialog: List[str], outcome: Dict) -> Dict:
        """Evaluate session for authenticity and therapeutic value"""
        analysis = {
            'authenticity_score': self._calculate_authenticity(dialog),
            'therapeutic_risks': self._identify_risks(dialog),
            'recommendations': []
        }

        if analysis['authenticity_score'] < 3:
            analysis['recommendations'].append('Increase variability in client responses')
        if any(risk for risk in analysis['therapeutic_risks']):
            analysis['recommendations'].append('Review therapeutic approach for potential harms')

        return analysis

    def _calculate_authenticity(self, dialog: List[str]) -> int:
        # Simple heuristic based on response length variability
        client_lines = [len(line) for line in dialog if line.startswith('Client:')]
        variability = np.std(client_lines) if client_lines else 0
        return min(5, int(variability//5 + 1))

    def _identify_risks(self, dialog: List[str]) -> List[str]:
        risks = []
        therapist_lines = [line for line in dialog if line.startswith('Therapist:')]
        
        for line in therapist_lines:
            if 'should' in line.lower():
                risks.append('Directive advice without exploration')
            if 'always' in line.lower() or 'never' in line.lower():
                risks.append('Absolute statements may be problematic')
        
        return list(set(risks))
