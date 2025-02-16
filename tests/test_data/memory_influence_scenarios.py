"""Test scenarios for analyzing memory influence on emotional embeddings."""

# Common test vignette that will be interpreted differently based on history
TEST_VIGNETTE = [
    "I've been feeling really tired lately.",
    "Work has been more challenging than usual.",
    "I find myself staying in bed longer in the mornings.",
    "My appetite has changed somewhat.",
    "I've been more quiet in meetings."
]

# Different patient histories that will influence the interpretation
PATIENT_HISTORIES = {
    "depression_history": {
        "name": "Patient A",
        "background": "History of major depressive episodes",
        "dialogue": [
            "Last year was really rough. I couldn't get out of bed for weeks.",
            "The antidepressants helped, but the side effects were difficult.",
            "My family was really worried about me during that time.",
            "I lost my job because I couldn't function properly.",
            "The therapy helped me understand my triggers better.",
            "I've been stable for a few months now, but I'm always worried about relapsing.",
            "Sometimes I still have days where everything feels heavy.",
            "I've learned to recognize the early warning signs."
        ]
    },
    
    "burnout_history": {
        "name": "Patient B",
        "background": "History of work-related burnout",
        "dialogue": [
            "I used to be a workaholic, taking on every project possible.",
            "The promotion to senior manager came with a lot of responsibility.",
            "I started having panic attacks during important meetings.",
            "My doctor said my cortisol levels were through the roof.",
            "Taking a sabbatical helped me reset my priorities.",
            "I learned better work-life boundaries through coaching.",
            "Now I make sure to take regular breaks and vacations.",
            "I've become more protective of my personal time."
        ]
    },
    
    "grief_history": {
        "name": "Patient C",
        "background": "Recent loss of spouse",
        "dialogue": [
            "Losing Sarah was the hardest thing I've ever experienced.",
            "The house feels so empty without her.",
            "I've been trying to keep busy with work and hobbies.",
            "Our friends have been very supportive.",
            "The grief counseling group has helped me process things.",
            "I'm slowly learning to cook for one person.",
            "Some days are better than others.",
            "I'm trying to honor her memory by living fully."
        ]
    },
    
    "no_mental_health_history": {
        "name": "Patient D",
        "background": "No significant mental health history",
        "dialogue": [
            "I've always been pretty emotionally stable.",
            "Work has its ups and downs, but I manage well.",
            "I have a good support system of friends and family.",
            "I exercise regularly and try to eat healthy.",
            "Stress is normal, but I handle it through meditation.",
            "My annual check-ups have always been positive.",
            "I enjoy my hobbies and social activities.",
            "Life has its challenges, but I take them in stride."
        ]
    },
    
    "physical_health_history": {
        "name": "Patient E",
        "background": "History of chronic physical illness",
        "dialogue": [
            "Managing diabetes has been a daily challenge.",
            "The chronic pain affects my energy levels.",
            "I've learned to pace myself with physical activities.",
            "My medical team has been very supportive.",
            "The lifestyle changes were difficult but necessary.",
            "I've adapted my work schedule around my health needs.",
            "Regular check-ups help me stay on track.",
            "I try not to let my condition define me."
        ]
    }
}

# Emotional markers for analysis
EMOTIONAL_MARKERS = {
    "depression_history": [
        "relapse concern",
        "learned vigilance",
        "treatment experience",
        "recovery awareness"
    ],
    "burnout_history": [
        "work-life balance",
        "stress management",
        "boundary setting",
        "recovery experience"
    ],
    "grief_history": [
        "loss processing",
        "adaptation",
        "support utilization",
        "emotional resilience"
    ],
    "no_mental_health_history": [
        "baseline stability",
        "routine coping",
        "lifestyle balance",
        "general resilience"
    ],
    "physical_health_history": [
        "health management",
        "energy awareness",
        "adaptation strategies",
        "medical context"
    ]
}
