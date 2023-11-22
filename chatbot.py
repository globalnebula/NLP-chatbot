import re
import json
from transformers import pipeline

def classify_tone(text):
    emotion_classifier = pipeline('sentiment-analysis')
    result = emotion_classifier(text)

    emotion = result[0]['label']
    return emotion

def load_responses():
    # Load response templates from a JSON file
    with open('data.json', 'r', encoding='utf-8') as file:
        responses = json.load(file)
    return responses

def check_patterns(user_input):
    # Define patterns and corresponding responses
    patterns = {
        r'\b(?:how\sare\syou|what\'s\sup)\b': "I'm here to assist you. How can I help?",
        r'\b(?:thank\syou|thanks|appreciate)\b': "You're welcome! If you need anything else, feel free to ask.",
        r'\b(?:tell\sme\sajoke|joke)\b': "Sure, here's a joke for you: [insert joke]",
    }

    # Check for patterns in the user input
    for pattern, response in patterns.items():
        if re.search(pattern, user_input, re.IGNORECASE):
            return response

    # No specific pattern found
    return None

def chatbot_response(user_input, responses):
    # Check for specific patterns
    pattern_response = check_patterns(user_input)
    if pattern_response:
        return pattern_response

    # Proceed with mood-based response
    tone = classify_tone(user_input)

    if tone in responses:
        return responses[tone]
    else:
        return "I'm not sure how to respond to that. Can you provide more details?"

# Example of using the functions
responses = load_responses()
user_input = input("How are you feeling today? ")
response = chatbot_response(user_input, responses)
print(response)