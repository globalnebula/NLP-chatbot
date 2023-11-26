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

def classify_tone(text, model, vectorizer):
    # Transform text using the vectorizer
    text_vectorized = vectorizer.transform([text])
    
    # Predict the emotion using the trained model
    emotion = model.predict(text_vectorized)[0]
    return emotion


def load_responses():
    # Load response templates from a JSON file
    with open('data.json', 'r', encoding="utf-8") as file:
        responses = json.load(file)
    return responses


def get_remedies(emotion):
    # Define remedies based on emotion
    remedies = {
        'joy': ['Take a walk in nature', 'Listen to uplifting music', 'Call a friend and share your joy'],
        'sadness': ['Reach out to a friend or family member', 'Engage in activities you enjoy', 'Consider talking to a professional'],
        'anger': ['Practice deep breathing exercises', 'Take a break to cool off', 'Express your feelings through writing'],
        'neutral': ['Take a moment to relax', 'Engage in a hobby', 'Plan something enjoyable for yourself'],
        'curiosity': ['Explore a new topic or hobby', 'Read an interesting article', 'Watch a documentary'],
        'anxiety': ['Practice mindfulness meditation', 'Focus on your breath', 'Create a calming routine']
    }

    return remedies.get(emotion, [])

def chatbot_response(user_input, responses, model, vectorizer):
    # Check for greetings
    if any(greeting in user_input.lower() for greeting in greetings):
        return "Hello! How can I assist you today?"

    # Check for farewells
    if any(farewell in user_input.lower() for farewell in farewells):
        return "Goodbye! Have a great day!"

    # Transform the user input using the vectorizer
    user_input_vectorized = vectorizer.transform([user_input])

    # Predict the emotion using the trained model
    emotion = model.predict(user_input_vectorized)[0]

    if emotion in responses:
        # Randomly select one response for the predicted emotion
        emotion_responses = responses[emotion]
        selected_response = random.choice(emotion_responses)

        # Get remedies for the emotion
        remedies = get_remedies(emotion)
        if remedies:
            selected_response += f"\n\nRemedies for {emotion.capitalize()}: {', '.join(remedies)}"

        return selected_response
    else:
        return "I'm not sure how to respond to that. Can you provide more details?"

def start_conversation(responses, model, vectorizer):
    print("Bot: Hello! How can I assist you today?")
    while True:
        user_input = input("You: ")

        # Check for farewell to end the conversation
        if any(farewell in user_input.lower() for farewell in farewells):
            print("Bot: Goodbye! Have a great day!")
            break

        response = chatbot_response(user_input, responses, model, vectorizer)
        print(f"Bot: {response}")


def main():
    # Data processing and model training
    df = pd.DataFrame(data)

    greeting_data = [{'text': f'{g} how are you feeling?', 'emotion': 'neutral'} for g in greetings]
    farewell_data = [{'text': f'{f}, it was nice chatting with you!', 'emotion': 'neutral'} for f in farewells]
    df = pd.concat([df, pd.DataFrame(greeting_data + farewell_data)], ignore_index=True)

    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['emotion'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train_vectors = vectorizer.fit_transform(X_train)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_vectors, y_train)

    responses = load_responses()
    start_conversation(responses, model, vectorizer)

if __name__ == "__main__":
    main()
