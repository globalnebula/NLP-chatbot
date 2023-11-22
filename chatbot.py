import re
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import random

# Expanded dataset
data = {
    'text': [
        'I feel happy', 'This is so sad', 'I am angry', 'Feeling scared', 'Surprised by the news', 'Disgusted by the behavior',
        'Excited about the upcoming event', 'Feeling content with life', 'Frustrated with work', 'Anxious about the future',
        'Overwhelmed with responsibilities', 'In awe of the beautiful scenery', 'Bored and looking for something to do',
        'Grateful for the support', 'Confused about the situation', 'Proud of the accomplishment', 'Enthusiastic about the project',
        'Worried about the health issue', 'Curious to learn new things', 'Nostalgic about the past', 'Hopeful for a positive outcome',
        'Cautious about the decision', 'Inspired by the creative work', 'Sympathetic towards others', 'Amused by the funny incident',
        'Determined to achieve the goal', 'Lonely and seeking companionship', 'Optimistic despite challenges', 'Skeptical about the plan',
        # Additional data for variety and coverage
        'Feeling fantastic today', 'Feeling down and out', 'Angry at the injustice', 'Thrilled about the opportunity',
        'Feeling blue', 'Eager to start a new project', 'Annoyed by the constant interruptions', 'Feeling on top of the world',
        'Indifferent about the situation', 'Excited to meet new people'
    ],
    'emotion': [
        'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust',
        'joy', 'contentment', 'anger', 'anxiety',
        'overwhelmed', 'awe', 'boredom', 'gratitude', 'confusion', 'pride', 'enthusiasm',
        'worry', 'curiosity', 'nostalgia', 'hope', 'caution', 'inspiration', 'sympathy', 'amusement',
        'determination', 'loneliness', 'optimism', 'skepticism',
        'joy', 'sadness', 'anger', 'joy', 'sadness', 'joy', 'anger', 'joy', 'neutral', 'joy'
    ]
}

df = pd.DataFrame(data)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['emotion'], random_state=42)

# Create a pipeline with a CountVectorizer and Multinomial Naive Bayes classifier
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

def classify_tone(text):
    # Predict the emotion using the trained model
    emotion = model.predict([text])[0]
    return emotion

def load_responses():
    # Load response templates from a JSON file
    with open('data.json', 'r', encoding="utf-8") as file:
        responses = json.load(file)
    return responses

def check_patterns(user_input, patterns):
    # Check for patterns in the user input
    for pattern, response in patterns.items():
        if re.search(pattern, user_input, re.IGNORECASE):
            return response

    # No specific pattern found
    return None

def chatbot_response(user_input, responses):
    # Check for specific patterns
    pattern_response = check_patterns(user_input, responses['patterns'])
    if pattern_response:
        return pattern_response

    # Proceed with mood-based response
    tone = classify_tone(user_input)

    if tone in responses:
        # Randomly select one response for the predicted emotion
        emotion_responses = responses[tone]
        selected_response = random.choice(emotion_responses)
        return selected_response
    else:
        return "I'm not sure how to respond to that. Can you provide more details?"

# Example of using the functions
responses = load_responses()
user_input = input("How are you feeling today? ")
response = chatbot_response(user_input, responses)
print(response)
