import tkinter as tk
from tkinter import scrolledtext
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import random
import re

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
        'Excited and curious', 'Feeling content and curious', 'Angry and frustrated', 'Joyful despite challenges', 'Curious about the future', 'I am gay'
    ],
    'emotion': [
        'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust',
        'joy', 'contentment', 'anger', 'anxiety',
        'overwhelmed', 'awe', 'boredom', 'gratitude', 'confusion', 'pride', 'enthusiasm',
        'worry', 'curiosity', 'nostalgia', 'hope', 'caution', 'inspiration', 'sympathy', 'amusement',
        'determination', 'loneliness', 'optimism', 'skepticism',
        'joy', 'sadness', 'anger', 'joy', 'sadness', 'joy', 'anger', 'joy', 'neutral', 'joy',
        'joy', 'joy', 'contentment', 'joy', 'surprise', 'contentment', 'anger', 'joy', 'neutral', 'curiosity',
        'joy', 'curiosity', 'anxiety', 'curiosity', 'sadness', 'joy', 'contentment', 'anger', 'joy', 'curiosity', 'sadness'
    ]
}

greetings = [
    'Hello', 'Hi', 'Hey', 'Greetings', 'Good morning', 'Good afternoon', 'Good evening'
]

farewells = [
    'Goodbye', 'Bye', 'Farewell', 'See you later', 'Take care', 'Adios'
]

class ChatbotUI:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
        self.responses = self.load_responses()
        self.window = tk.Tk()
        self.window.title("Chatbot UI")

        self.create_widgets()

    def load_responses(self):
        with open('data.json', 'r', encoding="utf-8") as file:
            responses = json.load(file)
        return responses

    def create_widgets(self):
        self.chat_history = scrolledtext.ScrolledText(self.window, width=50, height=20, wrap=tk.WORD, fg='black',bg="gray")
        self.chat_history.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        self.user_input_label = tk.Label(self.window, text="User Input:")
        self.user_input_label.grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)

        self.user_input_entry = tk.Entry(self.window, width=40, fg='black',bg="gray")
        self.user_input_entry.grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)

        self.send_button = tk.Button(self.window, text="Send", command=self.send_message)
        self.send_button.grid(row=2, column=1, padx=10, pady=5, sticky=tk.E)

        self.quit_button = tk.Button(self.window, text="Quit", command=self.window.quit)
        self.quit_button.grid(row=2, column=0, padx=10, pady=5, sticky=tk.W)

    def send_message(self):
        user_input = self.user_input_entry.get()
        self.user_input_entry.delete(0, tk.END)

        response = self.chatbot_response(user_input)
        self.update_chat_history(f"You: {user_input}\nBot: {response}")

    def chatbot_response(self, user_input):
        if any(farewell in user_input.lower() for farewell in farewells):
            return "Goodbye! Have a great day!"

        matched_pattern = self.match_pattern(user_input)
        if matched_pattern:
            return matched_pattern


        user_input_vectorized = self.vectorizer.transform([user_input])


        emotion = self.model.predict(user_input_vectorized)[0]

        if emotion in self.responses:
            emotion_responses = self.responses[emotion]
            selected_response = random.choice(emotion_responses)

            remedies = self.get_remedies(emotion)
            if remedies:
                selected_response += f"\n\nRemedies for {emotion.capitalize()}:"
                for remedy in remedies:
                    selected_response += f"\n- {remedy}"

            return selected_response
        else:
            return "I'm not sure how to respond to that. Can you provide more details?"

    def match_pattern(self, user_input):
        for pattern, response in self.responses.get("patterns", {}).items():
            if re.search(fr'\b{pattern}\b', user_input, re.IGNORECASE):
                return response
        return None



    def get_remedies(self, emotion):
        return self.responses.get("remedies", {}).get(emotion, [])

    def update_chat_history(self, message):
        self.chat_history.insert(tk.END, f"{message}\n\n")
        self.chat_history.yview(tk.END)

    def start(self):
        self.window.mainloop()

def main():
    df = pd.DataFrame(data)

    greeting_data = [{'text': f'{g} how are you feeling?', 'emotion': 'neutral'} for g in greetings]
    farewell_data = [{'text': f'{f}, it was nice chatting with you!', 'emotion': 'neutral'} for f in farewells]
    df = pd.concat([df, pd.DataFrame(greeting_data + farewell_data)], ignore_index=True)

    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['emotion'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train_vectors = vectorizer.fit_transform(X_train)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_vectors, y_train)

    chatbot_ui = ChatbotUI(model, vectorizer)
    chatbot_ui.start()

if __name__ == "__main__":
    main()