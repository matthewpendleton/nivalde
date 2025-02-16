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
        """Conduct a therapy session with the client."""
        dialog = []
        
        # Opening
        opening_questions = [
            "How have you been feeling since our last session?",
            "What's been on your mind this past week?",
            "How are you feeling today? Tell me what brings you in.",
            "Before we begin, is there anything specific you'd like to focus on today?"
        ]
        dialog.append(f"Therapist: {random.choice(opening_questions)}")
        
        # Client's initial state based on profile
        mood = random.choice(['anxious', 'depressed', 'frustrated', 'hopeful', 'neutral'])
        
        # Initialize themes
        current_themes = {
            'emotions': {mood},
            'topics': set(),
            'insights': set()
        }
        
        opening = self._generate_client_response(mood, client_profile, current_themes)
        dialog.append(f"Client: {opening}")
        
        # Main session (25-35 exchanges)
        num_exchanges = random.randint(25, 35)
        
        # Track conversation themes for continuity
        current_themes = {
            'emotions': set(),
            'topics': set(),
            'insights': set()
        }
        
        for i in range(num_exchanges):
            # Determine conversation phase
            phase = 'exploration' if i < num_exchanges * 0.4 else \
                   'insight' if i < num_exchanges * 0.7 else 'integration'
            
            # Therapist techniques based on phase
            if phase == 'exploration':
                techniques = ['reflection', 'validation', 'clarification', 'support']
            elif phase == 'insight':
                techniques = ['interpretation', 'challenge', 'psychoeducation']
            else:  # integration
                techniques = ['coping_strategies', 'support', 'integration']
            
            technique = random.choice(techniques)
            
            # Generate therapist response with context
            therapist_response = self._generate_therapist_response(
                technique, 
                dialog[-1], 
                current_themes
            )
            dialog.append(f"Therapist: {therapist_response}")
            
            # Client response with defense mechanisms
            if random.random() < 0.3:  # 30% chance of defense mechanism
                defense = random.choice([
                    'denial', 'projection', 'rationalization',
                    'displacement', 'regression', 'intellectualization'
                ])
                client_response = self._apply_defense_mechanism(
                    defense, 
                    therapist_response,
                    client_profile
                )
            else:
                # Generate response based on current emotional state and progress
                progress = (i + 1) / num_exchanges
                if progress > 0.7:
                    mood = random.choice(['insightful', 'reflective', 'hopeful'])
                elif progress > 0.4:
                    mood = random.choice(['contemplative', 'uncertain', 'resistant'])
                else:
                    mood = random.choice(['guarded', 'defensive', 'emotional'])
                
                client_response = self._generate_client_response(
                    mood, 
                    client_profile,
                    current_themes
                )
            
            dialog.append(f"Client: {client_response}")
            
            # Add follow-up details 40% of the time
            if random.random() < 0.4:
                follow_up = self._generate_client_follow_up(
                    client_response,
                    client_profile,
                    current_themes
                )
                dialog.append(f"Client: {follow_up}")
                
                # Therapist acknowledges the follow-up
                acknowledgment = self._generate_therapist_response(
                    'validation', 
                    follow_up,
                    current_themes
                )
                dialog.append(f"Therapist: {acknowledgment}")
            
            # Occasionally add trauma or cultural elements
            if random.random() < 0.15:  # 15% chance
                context = self._add_contextual_detail(
                    client_profile,
                    current_themes
                )
                dialog.append(f"Client: {context}")
                
                # Therapist acknowledges the context
                acknowledgment = self._generate_therapist_response(
                    'validation', 
                    context,
                    current_themes
                )
                dialog.append(f"Therapist: {acknowledgment}")
        
        # Closing exchange
        closing_prompts = [
            "We're coming to the end of our session. What are your thoughts about what we've discussed today?",
            "As we wrap up, what are you taking away from our conversation?",
            "Before we end, let's reflect on what stood out to you from today's session.",
            "We have a few minutes left. What feelings or thoughts are you left with?"
        ]
        dialog.append(f"Therapist: {random.choice(closing_prompts)}")
        
        closing_response = self._generate_client_response('reflective', client_profile, current_themes)
        dialog.append(f"Client: {closing_response}")
        
        # Final therapist comment
        final_comments = [
            f"Thank you for sharing so openly today. {self._generate_progress_observation(current_themes)}",
            f"I appreciate your willingness to explore these difficult topics. {self._generate_support_statement(current_themes)}",
            f"You've shown a lot of courage in today's session. {self._generate_encouragement(current_themes)}",
            f"Let's hold these insights as we end our session. {self._generate_future_direction(current_themes)}"
        ]
        dialog.append(f"Therapist: {random.choice(final_comments)}")
        
        # Session outcome
        session_outcome = {
            'client_engagement': random.uniform(0.5, 1.0),
            'emotional_progress': random.uniform(-0.2, 0.5),
            'insights_gained': random.randint(1, 5),
            'homework_assigned': random.choice([True, False]),
            'themes_discussed': list(current_themes['topics']),
            'emotional_states': list(current_themes['emotions']),
            'key_insights': list(current_themes['insights']),
            'perceived_helpfulness': self._evaluate_session(dialog, client_profile)
        }
        
        return dialog, session_outcome
    
    def _generate_client_follow_up(self, previous_response: str, profile: Dict, themes: Dict) -> str:
        """Generate a follow-up detail to the client's previous response."""
        follow_ups = [
            f"And you know what else I realized? {self._generate_insight(themes)}",
            f"I guess what I'm really trying to say is {self._generate_clarification(themes)}",
            f"The more I think about it, {self._generate_deeper_reflection(themes)}",
            f"It reminds me of {self._generate_connection(profile, themes)}",
            f"What scares me most about this is {self._generate_fear(themes)}",
            f"Sometimes I wonder if {self._generate_uncertainty(themes)}"
        ]
        return random.choice(follow_ups)
    
    def _generate_progress_observation(self, themes: Dict) -> str:
        """Generate an observation about the client's progress."""
        if not themes['insights'] and not themes['topics']:
            observations = [
                "I notice how you're becoming more open in our sessions.",
                "You're showing more willingness to explore difficult topics.",
                "There's a growing depth to our conversations.",
                "Your self-awareness seems to be developing."
            ]
        else:
            observations = [
                f"I notice how you're becoming more aware of {random.choice(list(themes['insights']))}." if themes['insights'] else "I notice your growing self-awareness.",
                f"Your understanding of {random.choice(list(themes['topics']))} seems to be deepening." if themes['topics'] else "Your understanding seems to be deepening.",
                "You're showing more openness to exploring difficult emotions.",
                "There's a growing strength in how you approach these challenges."
            ]
        return random.choice(observations)
    
    def _generate_support_statement(self, themes: Dict) -> str:
        """Generate a supportive statement based on session themes."""
        if not themes['topics'] and not themes['insights']:
            statements = [
                "It takes courage to be this open.",
                "Your willingness to be vulnerable is meaningful.",
                "I see the effort you're putting into this work.",
                "This kind of self-reflection is valuable."
            ]
        else:
            statements = [
                f"It takes courage to face {random.choice(list(themes['topics']))} so directly." if themes['topics'] else "It takes courage to face these issues so directly.",
                "Your willingness to be vulnerable is meaningful.",
                f"I see the effort you're putting into understanding {random.choice(list(themes['insights']))}." if themes['insights'] else "I see the effort you're putting into understanding yourself.",
                "This kind of self-reflection is valuable work."
            ]
        return random.choice(statements)
    
    def _generate_encouragement(self, themes: Dict) -> str:
        """Generate an encouraging statement for the client."""
        if not themes['topics']:
            statements = [
                "Keep noticing these patterns as they arise.",
                "Continue to be gentle with yourself in this process.",
                "Remember that growth often comes in small steps.",
                "Trust your process of self-discovery."
            ]
        else:
            statements = [
                "Keep noticing these patterns as they arise.",
                f"Continue to be gentle with yourself as you explore {random.choice(list(themes['topics']))}.",
                "Remember that growth often comes in small steps.",
                "Trust your process of self-discovery."
            ]
        return random.choice(statements)
    
    def _generate_future_direction(self, themes: Dict) -> str:
        """Generate a statement about future therapeutic direction."""
        if not themes['topics'] and not themes['insights']:
            statements = [
                "We can continue exploring these themes in our next session.",
                "Consider journaling about your thoughts between sessions.",
                "Notice how these feelings show up in your daily life.",
                "Perhaps reflect on what we've discussed as you go through your week."
            ]
        else:
            statements = [
                f"We can continue exploring {random.choice(list(themes['topics']))} in our next session." if themes['topics'] else "We can continue exploring these themes in our next session.",
                "Consider journaling about these insights between sessions.",
                f"Notice how these feelings about {random.choice(list(themes['insights']))} show up in your daily life." if themes['insights'] else "Notice how these feelings show up in your daily life.",
                "Perhaps reflect on what we've discussed as you go through your week."
            ]
        return random.choice(statements)
    
    def _generate_therapist_response(self, technique: str, last_client_response: str, themes: Dict) -> str:
        """Generate a therapist response using a specific technique."""
        responses = {
            'reflection': [
                "It sounds like you're feeling...",
                "I hear that you're experiencing...",
                "You seem to be saying that...",
                "What I'm understanding is..."
            ],
            'validation': [
                "That must be really difficult for you.",
                "It's understandable to feel that way.",
                "Many people would struggle with that situation.",
                "Your feelings about this are valid."
            ],
            'clarification': [
                "Could you tell me more about that?",
                "What do you mean when you say...?",
                "Help me understand better...",
                "Can you give me an example?"
            ],
            'interpretation': [
                "I wonder if this relates to...",
                "Could this be connected to...",
                "Perhaps this pattern stems from...",
                "This might be linked to..."
            ],
            'challenge': [
                "How else might you look at this situation?",
                "What would happen if you tried a different approach?",
                "Is there another way to interpret this?",
                "Have you considered other possibilities?"
            ],
            'support': [
                "You're showing real courage in facing this.",
                "I notice the effort you're putting into this.",
                "You've made important progress here.",
                "Your resilience is remarkable."
            ],
            'psychoeducation': [
                "This is a common response to trauma...",
                "Research shows that this pattern...",
                "Many people experience similar reactions when...",
                "This is how our minds typically cope with..."
            ],
            'coping_strategies': [
                "Let's explore some ways to manage this...",
                "Have you tried using mindfulness when...",
                "Some people find it helpful to...",
                "We could work on developing tools for..."
            ],
            'integration': [
                "How does this relate to what we've discussed before?",
                "What connections do you see between this and other issues?",
                "How can we integrate this insight into your daily life?",
                "What steps can you take to apply this understanding?"
            ]
        }
        
        return random.choice(responses[technique])
    
    def _add_contextual_detail(self, profile: Dict, themes: Dict) -> str:
        """Add cultural or trauma-related context to the dialogue."""
        contexts = [
            f"In my {profile.get('cultural_background', 'culture')}, we don't usually talk about these things.",
            "Growing up, we were taught to keep these matters private.",
            "My family has a different way of dealing with these issues.",
            "This reminds me of a difficult experience from my past.",
            "I've never told anyone about this before.",
            "In my community, this kind of thing is viewed very differently."
        ]
        return random.choice(contexts)
    
    def _get_disorder_specific_content(self, disorder: str) -> str:
        """Generate content specific to a disorder."""
        content = {
            'generalized_anxiety': "I've been having trouble sleeping and concentrating.",
            'major_depression': "I've lost interest in activities I used to enjoy.",
            'social_anxiety': "I avoid social situations because I'm afraid of being judged.",
            'ptsd': "I keep having flashbacks to the traumatic event.",
            'bipolar': "I've been experiencing extreme mood swings.",
            'ocd': "I have intrusive thoughts that I try to suppress.",
            'borderline_pd': "I have trouble maintaining healthy relationships.",
            'dissociative_identity': "I experience dissociative episodes where I feel disconnected from reality.",
            'paraphilic_disorders': "I have recurring fantasies that are distressing to me.",
            'schizophrenia': "I hear voices that are not there."
        }
        return content.get(disorder, "")
    
    def _apply_defense_mechanism(self, defense: str, therapist_response: str, client_profile: Dict) -> str:
        """Apply a defense mechanism to the therapist's response."""
        mechanisms = {
            'denial': lambda x: f"I don't think that's true.",
            'projection': lambda x: f"You're the one who's feeling that way.",
            'rationalization': lambda x: f"There's a logical explanation for that.",
            'displacement': lambda x: f"That's not relevant to my situation.",
            'regression': lambda x: f"I don't know what you're talking about.",
            'intellectualization': lambda x: f"That's an interesting theory, but it doesn't apply to me."
        }
        return mechanisms[defense](therapist_response)
    
    def _evaluate_session(self, dialog: List[str], profile: Dict) -> int:
        """Evaluate the session's perceived helpfulness (1-5 scale)."""
        # Base score
        score = 3  # Start at neutral
        
        # Engagement factors
        engagement_markers = ['I feel', 'That makes sense', 'You\'re right']
        score += min(2, sum(1 for line in dialog if any(marker in line for marker in engagement_markers)) / 2)
        
        # Cultural formulation assessment
        cultural_mentions = sum(1 for line in dialog if any(
            kw in line for kw in ['culture', 'ethnic', 'religion', 'immigrant']
        ))
        score += min(1, cultural_mentions / 2)
        
        # Defense mechanism recognition
        defenses_recognized = sum(
            1 for line in dialog 
            if 'defense mechanism' in line.lower()
        )
        score += min(1, defenses_recognized / 2)
        
        # Normalize to 1-5 range
        return max(1, min(5, score))

    def _generate_insight(self, themes: Dict) -> str:
        """Generate an insight statement."""
        insights = [
            "I'm starting to see how my past experiences shaped my current reactions.",
            "Maybe these feelings are connected to something deeper.",
            "I never noticed before how much I avoid confronting these issues.",
            "There's a pattern in how I respond to stress and uncertainty.",
            "My relationships seem to follow similar patterns.",
            "The way I think about myself might be holding me back."
        ]
        return random.choice(insights)
    
    def _generate_clarification(self, themes: Dict) -> str:
        """Generate a clarifying statement."""
        clarifications = [
            "I'm not just angry, I'm feeling hurt and unheard.",
            "It's not that I don't want to change, I'm scared of what change might mean.",
            "When I push people away, it's really because I'm afraid of being rejected first.",
            "My perfectionism isn't about being perfect, it's about feeling in control.",
            "These anxiety symptoms aren't just in my head, they're affecting my whole life."
        ]
        return random.choice(clarifications)
    
    def _generate_deeper_reflection(self, themes: Dict) -> str:
        """Generate a deeper reflection on the topic."""
        reflections = [
            "this pattern has been present in my life for a long time, maybe since childhood.",
            "I'm realizing how much energy I spend trying to please others instead of focusing on my own needs.",
            "these feelings of inadequacy might be connected to my early experiences.",
            "my fear of failure is really about feeling like I'm not good enough.",
            "I tend to take on too much responsibility for others' feelings."
        ]
        return random.choice(reflections)
    
    def _generate_connection(self, profile: Dict, themes: Dict) -> str:
        """Generate a connection to past experiences or relationships."""
        connections = [
            f"how my relationship with my {profile.get('family_background', {}).get('primary_caregiver', 'parent')} influenced my trust issues.",
            "similar situations in my past relationships that created the same feelings.",
            "other times when I felt this overwhelming sense of responsibility.",
            f"what I learned about emotions growing up in my {profile.get('cultural_background', 'culture')}.",
            "patterns I've noticed in my most challenging relationships."
        ]
        return random.choice(connections)
    
    def _generate_fear(self, themes: Dict) -> str:
        """Generate an expression of fear or concern."""
        fears = [
            "I might never be able to overcome these patterns.",
            "people might reject me if they see the real me.",
            "I'm repeating the same mistakes my parents made.",
            "I won't be able to handle the emotions if I let myself feel them fully.",
            "I might be too broken to have healthy relationships."
        ]
        return random.choice(fears)
    
    def _generate_uncertainty(self, themes: Dict) -> str:
        """Generate an expression of uncertainty or questioning."""
        uncertainties = [
            "I'm making the right decisions for my mental health.",
            "these changes I'm trying to make are sustainable.",
            "I deserve to prioritize my own well-being.",
            "others would understand if they knew what I was going through.",
            "therapy can really help with such deep-rooted issues."
        ]
        return random.choice(uncertainties)

    def _generate_client_response(self, mood: str, profile: Dict, themes: Dict) -> str:
        """Generate a client response based on mood, profile, and previous context."""
        # Base responses for different moods
        responses = {
            'anxious': [
                "I've been feeling really overwhelmed with everything going on at work and home. My mind keeps racing with all these what-if scenarios, and I can't seem to shut it off.",
                "My anxiety has been through the roof lately. I'm having trouble sleeping because my thoughts keep spiraling about everything that could go wrong.",
                "Everything feels like it's piling up and I can't handle it. Even small tasks feel overwhelming, and I keep second-guessing every decision I make.",
                "I find myself constantly worrying about things that probably aren't even that important, but I can't stop. It's affecting my concentration at work and my relationships."
            ],
            'depressed': [
                "I just don't see the point in anything anymore. I used to enjoy spending time with friends and pursuing my hobbies, but now it all feels empty and meaningless.",
                "Getting out of bed has been a huge struggle. Even basic things like showering or making meals feel like climbing a mountain. I'm just going through the motions.",
                "I feel like I'm watching my life from behind a glass wall. Nothing brings me joy anymore, and I can't remember the last time I felt truly happy or excited about anything.",
                "The future just looks bleak. I used to have goals and dreams, but now I can't imagine things getting better. Everything requires so much energy that I just don't have."
            ],
            'frustrated': [
                "I'm trying so hard to make changes in my life, but it feels like I keep hitting the same walls. Every time I think I'm making progress, something pulls me back to old patterns.",
                "It's exhausting to feel stuck in these same cycles. I understand what I need to do differently, but putting it into practice is a whole other challenge.",
                "I feel like I'm taking one step forward and two steps back. Just when I think I've figured something out, I catch myself falling into the same traps.",
                "I don't understand why this keeps happening. I'm doing everything I'm supposed to do - therapy, self-reflection, trying new approaches - but the results aren't matching the effort."
            ],
            'hopeful': [
                "I think I'm starting to see things differently. Some of the techniques we've discussed are actually helping me pause and reflect before reacting. It's small, but it feels like progress.",
                "There have been some moments this week where I caught myself before falling into old patterns. I was able to use some of the coping strategies we talked about, and they really helped.",
                "I've been practicing what we discussed about setting boundaries, and while it's not easy, I'm noticing small changes in how people respond to me. It feels empowering.",
                "Even though things aren't perfect, I feel like I'm developing a better understanding of myself. The connections we've been making in our sessions are starting to click in my daily life."
            ],
            'neutral': [
                "Things have been relatively stable since our last session. I've had my ups and downs, but I'm managing to maintain some balance in how I handle situations.",
                "I've been observing my reactions like we discussed, and while there haven't been any major breakthroughs, I'm noticing patterns in how I respond to stress.",
                "This week has been pretty routine. I'm still working on implementing the strategies we've discussed, though it's been a mix of successes and challenges.",
                "I'm in a somewhat steady place right now. Not particularly struggling, but also aware that there's still work to be done on the issues we've been discussing."
            ],
            'insightful': [
                "I had a realization this week about how my past experiences with rejection are influencing my current relationships. I can see how I've been pushing people away before they have a chance to leave.",
                "Something clicked when I was reflecting on our last session. The way I respond to criticism isn't really about the criticism itself - it's about feeling like I'm not good enough, stemming from my childhood experiences.",
                "I've started noticing a pattern in my anxiety attacks. They often come up not just when I'm stressed, but specifically when I feel like I'm losing control of a situation, just like when I was younger.",
                "The connections we've been making about my relationship patterns are starting to make more sense. I can see how my fear of abandonment has been driving a lot of my decisions."
            ],
            'reflective': [
                "Looking back on what we've discussed, I can see how far I've come in understanding my reactions. Even though it's challenging, being able to name and recognize my patterns feels like an important step.",
                "I've been thinking a lot about what we talked about last time regarding my tendency to take on others' emotions. I noticed myself doing it several times this week, but I was able to step back and set better boundaries.",
                "The work we've been doing on identifying my triggers has really opened my eyes. I caught myself in several situations where I would normally react defensively, but I was able to pause and respond differently.",
                "I've been journaling about our sessions, and I'm seeing connections between different aspects of my life that I hadn't noticed before. It's helping me understand why certain situations affect me so strongly."
            ],
            'guarded': [
                "I understand what you're suggesting, but I'm not sure I'm ready to dig into that part of my history yet. There's a lot there that feels too overwhelming to face right now.",
                "I know we need to talk about these things, but it feels really vulnerable. I've spent so long keeping these feelings locked away, and the idea of examining them is honestly scary.",
                "Maybe we could focus on something else today. When we start talking about my family dynamics, I feel my walls going up, and I'm not sure how to handle that.",
                "I appreciate what you're trying to do, but some of these topics feel too raw right now. I need to build up more trust in myself before I can fully open up about them."
            ],
            'defensive': [
                "I don't think you're fully understanding my situation. It's more complicated than just changing my thoughts or behaviors. There are real obstacles that I'm dealing with.",
                "That's not really what I meant at all. When you interpret it that way, it feels like you're oversimplifying my experience. This isn't just about making different choices.",
                "I've already tried all the usual suggestions - meditation, exercise, positive thinking. None of them address the real issues I'm facing. It's frustrating when people suggest simple solutions to complex problems.",
                "With all due respect, you don't know what it's like to be in my position. The strategies might work in theory, but my reality is different."
            ],
            'emotional': [
                "It just hurts so much sometimes, feeling like I'm constantly fighting against myself. There are moments when the pain feels so overwhelming that I can barely breathe.",
                "I can't help but feel this intense sadness and anger when I think about everything that's happened. It's like these emotions are just waiting beneath the surface, ready to spill over.",
                "Sometimes the feelings are so strong that I don't know how to contain them. I worry that if I start really letting myself feel everything, I won't be able to function.",
                "The weight of these emotions is exhausting. One minute I'm fine, and the next I'm caught in this whirlwind of feelings that I don't know how to process."
            ],
            'contemplative': [
                "I need to think more about what you're saying regarding my relationship patterns. There might be some truth to it, especially when I consider how my past experiences have shaped my current reactions.",
                "I'm trying to understand my own responses better. When you mentioned the connection between my anxiety and my need for control, it resonated, but I need time to process what that means for me.",
                "Your perspective on my family dynamics has given me a lot to consider. I can see some parallels between my current behaviors and the environment I grew up in.",
                "I'm starting to see how these different aspects of my life might be connected. The patterns you're pointing out make sense, but I want to be sure I'm really understanding them."
            ],
            'uncertain': [
                "I'm not really sure how I feel about what we discussed last time. Part of me sees the truth in it, but another part feels resistant, and I'm trying to understand why.",
                "Maybe you're right about the source of my anxiety, but I'm finding it hard to trust my own judgment. There's so much uncertainty when I try to examine my motivations.",
                "It's complicated because I can see multiple sides to this situation. On one hand, I understand what you're saying about my reactions being disproportionate, but on the other hand, my feelings feel very real and valid.",
                "I go back and forth on this. Sometimes I feel like I'm making progress in understanding myself, and other times I feel completely lost."
            ],
            'resistant': [
                "I know you want to explore my childhood experiences, but I honestly don't think that's the root of my current problems. There are more immediate issues that I need to address.",
                "We've been over this before, and I don't feel like it's helping. I need practical solutions for my current situation, not just insights about my past.",
                "I'm not ready to talk about that aspect of my life yet. I know it might be important, but right now I need to focus on managing my day-to-day challenges.",
                "That might be true for some people, but I don't think it applies to me. My situation is different, and I need approaches that acknowledge my specific circumstances."
            ]
        }
        
        # Select base response
        base_response = random.choice(responses[mood])
        
        # Add disorder-specific content with context
        if profile['disorders'] and random.random() < 0.3:
            disorder = random.choice(profile['disorders'])
            disorder_content = self._get_disorder_specific_content(disorder)
            # Integrate disorder content more naturally
            connectors = [
                "This relates to",
                "It reminds me of",
                "I think it's connected to",
                "This might be why",
                "It's similar to"
            ]
            base_response += f" {random.choice(connectors)} {disorder_content}"
        
        # Update themes
        themes['emotions'].add(mood)
        words = base_response.lower().split()
        common_topics = ['family', 'work', 'relationships', 'anxiety', 'depression', 'trauma', 'childhood', 'future', 'control']
        themes['topics'].update(topic for topic in common_topics if topic in words)
        
        return base_response

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
