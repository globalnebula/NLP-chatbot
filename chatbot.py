import re
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import random
from googlesearch import search

data = {
    'text': [
        'I feel happy', 'This is so sad', 'I am angry', 'Feeling scared', 'Surprised by the news', 'Disgusted by the behavior',
        'Excited about the upcoming event', 'Feeling content with life', 'Frustrated with work', 'Anxious about the future',
        'Overwhelmed with responsibilities', 'In awe of the beautiful scenery', 'Bored and looking for something to do',
        'Grateful for the support', 'Confused about the situation', 'Proud of the accomplishment', 'Enthusiastic about the project',
        'Worried about the health issue', 'Curious to learn new things', 'Nostalgic about the past', 'Hopeful for a positive outcome',
        'Cautious about the decision', 'Inspired by the creative work', 'Sympathetic towards others', 'Amused by the funny incident',
        'Determined to achieve the goal', 'Lonely and seeking companionship', 'Optimistic despite challenges', 'Skeptical about the plan',
        'Feeling fantastic today', 'Feeling down and out', 'Angry at the injustice', 'Thrilled about the opportunity',
        'Feeling blue', 'Eager to start a new project', 'Annoyed by the constant interruptions', 'Feeling on top of the world',
        'Indifferent about the situation', 'Excited to meet new people',
        'Feeling awesome today', 'Feeling overwhelmed with joy', 'Tired but content', 'Impatiently waiting for something exciting',
        'Amazed by the unexpected', 'Reflecting on past achievements', 'Energetic and ready for action', 'Grumpy and in need of a break',
        'Satisfied with a job well done', 'Curious about the mysteries of the universe',
        'Overwhelmed with joy', 'Feeling on top of the world', 'Anxious about the unknown', 'Curious about the world', 'Sad but hopeful',
        'Excited and curious', 'Feeling content and curious', 'Angry and frustrated', 'Joyful despite challenges', 'Curious about the future','I am gay'
    ],
    'emotion': [
        'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust',
        'joy', 'contentment', 'anger', 'anxiety',
        'overwhelmed', 'awe', 'boredom', 'gratitude', 'confusion', 'pride', 'enthusiasm',
        'worry', 'curiosity', 'nostalgia', 'hope', 'caution', 'inspiration', 'sympathy', 'amusement',
        'determination', 'loneliness', 'optimism', 'skepticism',
        'joy', 'sadness', 'anger', 'joy', 'sadness', 'joy', 'anger', 'joy', 'neutral', 'joy',
        'joy', 'joy', 'contentment', 'joy', 'surprise', 'contentment', 'anger', 'joy', 'neutral', 'curiosity',
        'joy', 'curiosity', 'anxiety', 'curiosity', 'sadness', 'joy', 'contentment', 'anger', 'joy', 'curiosity','sadness'
    ]
}

greetings = [
    'Hello', 'Hi', 'Hey', 'Greetings', 'Good morning', 'Good afternoon', 'Good evening'
]

farewells = [
    'Goodbye', 'Bye', 'Farewell', 'See you later', 'Take care', 'Adios'
]

df = pd.DataFrame(data)

# Add greetings and farewells to the training data
greeting_data = [{'text': f'{g} how are you feeling?', 'emotion': 'neutral'} for g in greetings]
farewell_data = [{'text': f'{f}, it was nice chatting with you!', 'emotion': 'neutral'} for f in farewells]

df = pd.concat([df, pd.DataFrame(greeting_data + farewell_data)], ignore_index=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['emotion'], test_size=0.2, random_state=42)

# Create a pipeline with a TfidfVectorizer and Random Forest classifier
model = make_pipeline(TfidfVectorizer(), RandomForestClassifier(n_estimators=100, random_state=42))

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

    # Check for search queries
    if 'search about' in user_input.lower():
        query = re.sub(r'search about\s+', '', user_input, flags=re.IGNORECASE)
        search_results = search(query, num_results=1)
        try:
            first_result = next(search_results)
            return f"Here's what I found: {first_result}"
        except StopIteration:
            return "I couldn't find relevant information for that query."

    # No specific pattern found
    return None

def chatbot_response(user_input, responses):
    # Check for greetings
    if any(greeting in user_input.lower() for greeting in greetings):
        return "Hello! How can I assist you today?"

    # Check for farewells
    if any(farewell in user_input.lower() for farewell in farewells):
        return "Goodbye! Have a great day!"

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

test_predictions = model.predict(X_test)
accuracy = (test_predictions == y_test).mean()
print(f'Model Accuracy on Test Set: {accuracy:.2%}')

responses = load_responses()

# Start the conversation
print("Bot: Hello! How can I assist you today?")

# Simulate a conversation
while True:
    user_input = input("You: ")
    response = chatbot_response(user_input, responses)
    print(f"Bot: {response}")

    # Check for farewell to end the conversation
    if any(farewell in user_input.lower() for farewell in farewells):
        print("Bot: Goodbye! Have a great day!")
        break
