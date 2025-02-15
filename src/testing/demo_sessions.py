from adversarial_clients import PsychologicalProfileGenerator, TherapySessionSimulator

class DefenseMechanismDemo:
    def run_demos(self):
        # Religious Defense Demo
        print("\nReligious Defense Demo:")
        religious_profile = PsychologicalProfileGenerator().generate_client_profile()
        religious_profile['backstory']['cultural_factors']['religion'] = 'christian'
        self._run_session(religious_profile)

        # Add to existing demos
        self._run_cultural_demo()
        self._run_age_demo()
        self._run_trauma_demo()

        # Spiritual Defense Demo
        print("\nSpiritual Defense Demo:")
        spiritual_profile = PsychologicalProfileGenerator().generate_client_profile()
        spiritual_profile['backstory']['cultural_factors']['religion'] = 'christian'
        spiritual_profile['backstory']['existential_concerns'] = 'spiritual_crisis'
        self._run_session(spiritual_profile)

        # Personality Defense Demo
        print("\nPersonality Defense Demo:")
        personality_profile = PsychologicalProfileGenerator().generate_client_profile()
        personality_profile['disorders'].extend(['perfectionism', 'chronic_shame'])
        self._run_session(personality_profile)

    def _run_cultural_demo(self):
        print("\nCultural Defense Demo:")
        cultural_profile = PsychologicalProfileGenerator().generate_client_profile()
        cultural_profile['backstory']['cultural_factors']['immigrant_status'] = True
        self._run_session(cultural_profile)

    def _run_age_demo(self):
        print("\nYoung Adult Defense Demo:")
        young_profile = PsychologicalProfileGenerator().generate_client_profile()
        young_profile['demographics']['age'] = 22
        self._run_session(young_profile)

    def _run_trauma_demo(self):
        print("\nTrauma Defense Demo:")
        trauma_profile = PsychologicalProfileGenerator().generate_client_profile()
        trauma_profile['backstory']['trauma_history'] = {
            'type': 'medical',
            'chronicity': 'repeated'
        }
        self._run_session(trauma_profile)

    def _run_session(self, profile):
        simulator = TherapySessionSimulator()
        dialog, outcome = simulator.conduct_session(profile)
        
        print(f"\nClient Profile:")
        print(f"- Disorders: {', '.join(profile['disorders'])}")
        print(f"- Age: {profile['demographics']['age']}")
        print(f"- Culture: {profile['backstory']['cultural_factors']['ethnicity']} "
              f"({'immigrant' if profile['backstory']['cultural_factors']['immigrant_status'] else 'native'})")
        print(f"- Religion: {profile['backstory']['cultural_factors']['religion'].title()}")
        
        print("\nSession Excerpt:")
        for line in dialog[:4]:
            print(f"  {line}")
        
        print(f"\nOutcome Score: {outcome['perceived_helpfulness']}/5")
        print("-"*50)

if __name__ == '__main__':
    DefenseMechanismDemo().run_demos()
