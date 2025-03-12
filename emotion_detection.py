
import numpy as np
import pandas as pd
import neattext.functions as nfx
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
data = {
    "text": [
        # Happy
        "I am very happy today!", "This is the best day of my life!",
        "I can't stop smiling, everything is great.", "Such a wonderful experience!",
        "Feeling amazing after achieving my goals.", "I love my family and friends.",
        "Life is so beautiful!", "I got my dream job, so happy!",
        "The sun is shining, and I feel awesome.", "This meal tastes fantastic!",

        # Sad
        "I feel so lonely and lost.", "Nothing seems to make me happy anymore.",
        "I just want to cry all day.", "Everything is falling apart.",
        "I miss my best friend so much.", "It's hard to move on after such loss.",
        "Today was a terrible day.", "I feel empty inside.",
        "I don't know how to handle this sadness.", "My heart is broken.",

        # Angry
        "I am so mad right now!", "Why does this always happen to me?",
        "I can't believe how unfair this is!", "This is so frustrating!",
        "People need to stop being so rude.", "I absolutely hate this!",
        "Everything is annoying today.", "Stop wasting my time!",
        "I can't take this anymore.", "Why are people so inconsiderate?",

        # Fear
        "I am scared of the dark.", "I feel anxious about tomorrow's meeting.",
        "I don't know what to do, I'm terrified.", "This place gives me chills.",
        "I am afraid something bad will happen.", "My heart is racing with fear.",
        "I don't want to be alone right now.", "The thought of failing scares me.",
        "I feel uneasy about this situation.", "What if things go wrong?",

        # Surprise
        "Wow! I did not expect this at all!", "This is the most unexpected news ever.",
        "I can't believe what just happened!", "You seriously got me a gift?",
        "This is a huge surprise!", "No way! This is unbelievable!",
        "Wait, what?! That was so unexpected!", "I’m speechless, this is shocking!",
        "I never saw this coming!", "I am genuinely surprised right now.",

        # Neutral
        "I feel okay, nothing special.", "Today was just another normal day.",
        "Nothing exciting happened today.", "I am just going with the flow.",
        "I'm neither happy nor sad.", "I feel indifferent about this.",
        "Just another usual day at work.", "Everything is just fine, nothing more.",
        "I don’t have much to say about today.", "It’s an ordinary day.",

        # Disgust
        "That was absolutely disgusting!", "I feel sick just thinking about it.",
        "I can’t believe people enjoy that.", "This smells so awful!",
        "I feel grossed out by this.", "That food was terrible!",
        "Why do people act like this?", "This place is so unhygienic.",
        "I can't stand this behavior.", "I am so repulsed right now.",

        # Love
        "I love spending time with you.", "You make my heart so full.",
        "I can’t imagine life without you.", "This is pure happiness, I love it!",
        "I feel so connected to you.", "My heart is filled with love.",
        "Every moment with you is special.", "Love is the most beautiful feeling.",
        "You mean the world to me.", "I cherish every second with you."
    ],
    "emotion": [
        "happy", "happy", "happy", "happy", "happy",
        "happy", "happy", "happy", "happy", "happy",
        
        "sad", "sad", "sad", "sad", "sad",
        "sad", "sad", "sad", "sad", "sad",
        
        "angry", "angry", "angry", "angry", "angry",
        "angry", "angry", "angry", "angry", "angry",
        
        "fear", "fear", "fear", "fear", "fear",
        "fear", "fear", "fear", "fear", "fear",
        
        "surprise", "surprise", "surprise", "surprise", "surprise",
        "surprise", "surprise", "surprise", "surprise", "surprise",
        
        "neutral", "neutral", "neutral", "neutral", "neutral",
        "neutral", "neutral", "neutral", "neutral", "neutral",
        
        "disgust", "disgust", "disgust", "disgust", "disgust",
        "disgust", "disgust", "disgust", "disgust", "disgust",
        
        "love", "love", "love", "love", "love",
        "love", "love", "love", "love", "love"
    ]
}
df = pd.DataFrame(data)
df["clean_text"] = df["text"].apply(nfx.remove_special_characters)  # Remove special characters
df["clean_text"] = df["clean_text"].apply(nfx.remove_stopwords)  # Remove stop words
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_text"])
y = df["emotion"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
joblib.dump(model, "emotion_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
def predict_emotion(text):
    loaded_model = joblib.load("emotion_model.pkl")
    loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    text_clean = nfx.remove_special_characters(text)
    text_clean = nfx.remove_stopwords(text_clean)
    text_vectorized = loaded_vectorizer.transform([text_clean])
    return loaded_model.predict(text_vectorized)[0]
while True:
    user_input = input("\nEnter a sentence (or type 'exit' to stop): ")
    if user_input.lower() == "exit":
        print("Exiting emotion detection.")
        break
    predicted_emotion = predict_emotion(user_input)
    print(f"Predicted Emotion: {predicted_emotion}")

