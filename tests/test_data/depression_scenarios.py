"""Test scenarios for depression vs non-depression emotional states.

This module contains carefully crafted scenarios to test the emotional embedding system's
ability to distinguish between depressed and non-depressed emotional patterns.
"""

# Depression Scenarios - Each represents a different manifestation of depression
DEPRESSION_SCENARIOS = [
    {
        "id": "high_functioning_depression",
        "persona": {
            "name": "Sarah",
            "age": 32,
            "occupation": "Software Engineer",
            "background": "Successful career, appears happy externally, struggles internally"
        },
        "dialogue": [
            "I got another promotion at work today. Everyone was congratulating me, but I felt... empty inside.",
            "Sometimes I wonder if any of this matters. I'm doing everything right, but I feel nothing.",
            "I keep up with my work, my apartment is clean, I exercise... but it's all just going through the motions.",
            "The worst part is that no one can tell. They see me smiling and think everything's fine.",
            "I'm so tired of pretending. Not just tired physically, but tired in my soul."
        ]
    },
    {
        "id": "grief_induced_depression",
        "persona": {
            "name": "Michael",
            "age": 45,
            "occupation": "Teacher",
            "background": "Lost spouse two years ago, struggling with prolonged grief"
        },
        "dialogue": [
            "The students keep asking why I don't tell jokes like I used to. I just... can't anymore.",
            "Every morning I wake up and for a split second, I forget she's gone. Then it hits me all over again.",
            "I know I should be moving forward, but I feel stuck. Like I'm frozen in time while everyone else moves on.",
            "The counselor says it's normal, but this doesn't feel normal. Nothing feels normal anymore.",
            "I used to love teaching, but now it's just another day to get through."
        ]
    },
    {
        "id": "postpartum_depression",
        "persona": {
            "name": "Elena",
            "age": 29,
            "occupation": "New Mother",
            "background": "First-time mother struggling with postpartum depression"
        },
        "dialogue": [
            "Everyone says this should be the happiest time of my life. Why can't I feel that joy?",
            "I love my baby, I do. But sometimes I look at her and feel... disconnected, like I'm watching someone else's life.",
            "The guilt is overwhelming. She deserves a better mother than what I can be right now.",
            "I'm afraid to tell anyone how I feel. What kind of mother doesn't feel instant love and happiness?",
            "I keep waiting for these feelings to pass, but they just get heavier each day."
        ]
    },
    {
        "id": "treatment_resistant_depression",
        "persona": {
            "name": "James",
            "age": 41,
            "occupation": "Accountant",
            "background": "Long-term depression, multiple failed treatments"
        },
        "dialogue": [
            "I've tried everything - medication, therapy, meditation, exercise. Nothing seems to work.",
            "The doctors keep saying to be patient, but it's been years. How much longer am I supposed to wait?",
            "Sometimes I wonder if this is just who I am now. If this grey fog is my normal.",
            "People don't understand why I can't 'just get better.' As if I haven't been trying every single day.",
            "I'm tired of trying new treatments only to have my hopes crushed again and again."
        ]
    }
]

# Non-Depression Scenarios - Various emotional states without depression
NON_DEPRESSION_SCENARIOS = [
    {
        "id": "life_transition_adjustment",
        "persona": {
            "name": "Alex",
            "age": 28,
            "occupation": "Graduate Student",
            "background": "Moving to a new city for PhD program"
        },
        "dialogue": [
            "It's definitely challenging being in a new city, but I'm excited about all the possibilities.",
            "I miss my friends back home, but I've been joining study groups and meeting new people.",
            "The program is intense, but in a good way. I feel like I'm growing every day.",
            "Sometimes I feel overwhelmed, but then I remember why I chose this path.",
            "Even on tough days, I can see the light at the end of the tunnel."
        ]
    },
    {
        "id": "work_life_balance",
        "persona": {
            "name": "Maya",
            "age": 35,
            "occupation": "Restaurant Owner",
            "background": "Managing a successful business while raising family"
        },
        "dialogue": [
            "Running a restaurant while raising kids is crazy sometimes, but I wouldn't have it any other way.",
            "Sure, there are days when I feel stretched thin, but seeing both my business and kids grow is worth it.",
            "I've learned to take breaks when I need them. Self-care isn't selfish, it's necessary.",
            "The community we've built here, both with customers and staff, feels like an extended family.",
            "Life is beautifully chaotic, and I'm grateful for every moment of it."
        ]
    },
    {
        "id": "recovery_journey",
        "persona": {
            "name": "David",
            "age": 38,
            "occupation": "Physical Therapist",
            "background": "Recovered from major injury, found new purpose"
        },
        "dialogue": [
            "The accident changed my perspective on everything. Now I help others through their own recovery.",
            "Every patient's progress reminds me of my own journey. It's incredibly fulfilling.",
            "Sure, some days are harder than others, but that's just part of being human.",
            "I've learned that setbacks are just setups for comebacks. It's all about perspective.",
            "Life threw me a curveball, but it led me to where I was meant to be."
        ]
    },
    {
        "id": "empty_nester_adjustment",
        "persona": {
            "name": "Linda",
            "age": 52,
            "occupation": "Artist",
            "background": "Recently became empty nester, rediscovering personal passions"
        },
        "dialogue": [
            "It was emotional when the kids first left, but now I'm rediscovering who I am beyond being a mom.",
            "I've started painting again after 20 years. It feels like reuniting with an old friend.",
            "My husband and I are dating each other again. It's like a second honeymoon phase.",
            "I miss the kids, but seeing them build their own lives brings me so much joy.",
            "This new chapter has brought unexpected opportunities for growth and joy."
        ]
    }
]

# Additional metadata for emotional analysis
EMOTIONAL_MARKERS = {
    "depression": {
        "common_phrases": [
            "feel nothing",
            "tired",
            "empty",
            "stuck",
            "overwhelmed",
            "hopeless",
            "worthless",
            "numb",
            "disconnected",
            "heavy"
        ],
        "temporal_patterns": [
            "persistent",
            "chronic",
            "ongoing",
            "constant",
            "never-ending"
        ],
        "cognitive_patterns": [
            "negative self-talk",
            "catastrophizing",
            "black-and-white thinking",
            "overgeneralization",
            "rumination"
        ]
    },
    "non_depression": {
        "common_phrases": [
            "excited",
            "grateful",
            "growing",
            "learning",
            "hopeful",
            "connected",
            "fulfilled",
            "challenged",
            "supported",
            "optimistic"
        ],
        "temporal_patterns": [
            "temporary",
            "transitional",
            "phase",
            "moment",
            "period"
        ],
        "cognitive_patterns": [
            "balanced perspective",
            "problem-solving",
            "growth mindset",
            "adaptability",
            "resilience"
        ]
    }
}
