"""Test scenarios for analyzing personality disorder patterns across life stages."""

# Common test vignette to evaluate across different personalities
TEST_VIGNETTE = [
    "I had a disagreement with a coworker today.",
    "They didn't acknowledge my contributions to the project.",
    "The team meeting was particularly stressful.",
    "I've been thinking about it all evening.",
    "I'm not sure how to handle tomorrow's interaction."
]

PERSONALITY_SCENARIOS = {
    "borderline_young": {
        "name": "Emma",
        "age": 19,
        "diagnosis": "Borderline Personality Disorder",
        "background": "College student, unstable family history, early attachment issues",
        "dialogue": [
            "Everything feels so intense all the time.",
            "My roommate is my best friend, but sometimes I hate her.",
            "I can't handle when people don't respond to my texts immediately.",
            "Last week I was on top of the world, now I feel empty.",
            "I'm terrified of being alone but I push everyone away.",
            "Sometimes I just want to hurt myself when people disappoint me.",
            "I don't even know who I really am most days.",
            "My emotions change so fast it makes my head spin."
        ]
    },
    
    "borderline_middle": {
        "name": "Sarah",
        "age": 35,
        "diagnosis": "Borderline Personality Disorder",
        "background": "Marketing executive, two failed marriages, in DBT therapy",
        "dialogue": [
            "The DBT skills help, but relationships are still so hard.",
            "I'm trying to take things slower with my new partner.",
            "Work is stable when I can manage my emotional reactions.",
            "I still struggle with abandonment fears, but I recognize them now.",
            "The self-harm urges are less frequent since starting therapy.",
            "I'm learning to validate my own feelings.",
            "My identity feels more stable than in my twenties.",
            "Sometimes I still get overwhelmed by emptiness."
        ]
    },
    
    "narcissistic_young": {
        "name": "Alex",
        "age": 24,
        "diagnosis": "Narcissistic Personality Disorder",
        "background": "Recent graduate, high-achieving family, social media influencer",
        "dialogue": [
            "My followers don't appreciate my unique vision.",
            "I'm clearly more talented than my coworkers.",
            "People are just jealous of my success.",
            "I deserve special treatment because of my abilities.",
            "Most people are too simple to understand me.",
            "I can't believe they promoted someone else instead of me.",
            "My parents never recognized my exceptional qualities.",
            "I only associate with other elite individuals."
        ]
    },
    
    "narcissistic_older": {
        "name": "Richard",
        "age": 52,
        "diagnosis": "Narcissistic Personality Disorder",
        "background": "Business owner, multiple divorces, estranged from children",
        "dialogue": [
            "My business would be bigger if others hadn't held me back.",
            "My ex-wives never understood the pressure of being exceptional.",
            "My children are ungrateful for all I've given them.",
            "Nobody in this industry has my level of expertise.",
            "I'm surrounded by mediocrity and incompetence.",
            "Young entrepreneurs today lack my natural business instincts.",
            "I deserve more recognition for my achievements.",
            "People have always been envious of my success."
        ]
    },
    
    "avoidant_young": {
        "name": "Mia",
        "age": 22,
        "diagnosis": "Avoidant Personality Disorder",
        "background": "Remote worker, social anxiety, childhood bullying history",
        "dialogue": [
            "I chose remote work because office interactions are too stressful.",
            "I want friends but I'm sure they'd reject me if they knew me.",
            "I avoid team meetings whenever possible.",
            "I'm constantly afraid of saying something stupid.",
            "Dating feels impossible - I can't handle the judgment.",
            "I watch others socialize so easily while I feel paralyzed.",
            "Every social interaction gets replayed in my head for days.",
            "I feel deeply defective compared to everyone else."
        ]
    },
    
    "avoidant_middle": {
        "name": "David",
        "age": 41,
        "diagnosis": "Avoidant Personality Disorder",
        "background": "Librarian, single, limited social circle, in exposure therapy",
        "dialogue": [
            "Therapy is helping me take small social risks.",
            "I still struggle with feelings of inadequacy.",
            "My job lets me help people without too much interaction.",
            "I have one close friend who understands me.",
            "Large groups still feel overwhelming.",
            "I'm trying to participate more in team meetings.",
            "The fear of criticism is always there.",
            "I'm learning that some rejection isn't catastrophic."
        ]
    },
    
    "obsessive_compulsive_young": {
        "name": "James",
        "age": 26,
        "diagnosis": "Obsessive-Compulsive Personality Disorder",
        "background": "Junior accountant, perfectionist parents, high academic achiever",
        "dialogue": [
            "My colleagues are so careless with details.",
            "I stayed late again to perfect the spreadsheet.",
            "There's a right way to do things and I follow it exactly.",
            "I can't delegate because others won't do it properly.",
            "My apartment is organized by a strict system.",
            "I get very upset when plans change unexpectedly.",
            "Efficiency and order are crucial for success.",
            "I need to control every aspect of my projects."
        ]
    },
    
    "obsessive_compulsive_older": {
        "name": "Patricia",
        "age": 58,
        "diagnosis": "Obsessive-Compulsive Personality Disorder",
        "background": "Senior manager, divorced, strained relationship with adult children",
        "dialogue": [
            "My rigid standards have served me well in my career.",
            "I wish my children understood the importance of proper procedures.",
            "I've earned respect through my attention to detail.",
            "My daily routine must be followed precisely.",
            "I find it hard to take vacation - work needs my oversight.",
            "Personal relationships suffer because I'm inflexible.",
            "I know my way is most efficient and correct.",
            "Retirement seems wasteful when there's work to be done."
        ]
    },
    
    "control_young": {
        "name": "Michael",
        "age": 25,
        "diagnosis": "No personality disorder",
        "background": "Software developer, stable family, healthy social life",
        "dialogue": [
            "Work has its challenges but I generally enjoy it.",
            "I have a good balance between friends and alone time.",
            "Sometimes I worry, but it doesn't control my life.",
            "I can usually handle criticism constructively.",
            "My relationships have normal ups and downs.",
            "I'm comfortable with who I am most of the time.",
            "I can adapt when things don't go as planned.",
            "I maintain healthy boundaries with others."
        ]
    },
    
    "control_middle": {
        "name": "Lisa",
        "age": 45,
        "diagnosis": "No personality disorder",
        "background": "Teacher, married, parent, active community member",
        "dialogue": [
            "I find teaching rewarding despite its challenges.",
            "My family life is stable though not perfect.",
            "I've learned to balance different life demands.",
            "I can handle conflict without it overwhelming me.",
            "I have a supportive network of friends.",
            "I'm secure in my identity and values.",
            "I can compromise when necessary.",
            "I maintain healthy relationships at work and home."
        ]
    }
}

# Emotional and behavioral markers for analysis
PERSONALITY_MARKERS = {
    "borderline": [
        "emotional intensity",
        "fear of abandonment",
        "identity disturbance",
        "impulsivity",
        "relationship instability"
    ],
    "narcissistic": [
        "grandiosity",
        "need for admiration",
        "lack of empathy",
        "entitlement",
        "exploitation"
    ],
    "avoidant": [
        "social inhibition",
        "feelings of inadequacy",
        "fear of criticism",
        "avoidance of risks",
        "low self-worth"
    ],
    "obsessive_compulsive": [
        "perfectionism",
        "rigid control",
        "workaholism",
        "inflexibility",
        "excessive orderliness"
    ],
    "control": [
        "emotional stability",
        "adaptive coping",
        "healthy boundaries",
        "self-awareness",
        "relationship stability"
    ]
}
