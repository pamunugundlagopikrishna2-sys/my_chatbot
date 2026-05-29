# chatbot.py - main file

import nltk
import string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK data download
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# FAQ DATA
faqs = [
    {"question": "shop timing what", "answer": "Our shop is open daily from 9 AM to 9 PM."},
    {"question": "delivery time how long", "answer": "Delivery takes 3-5 days."},
    {"question": "return policy", "answer": "You can return within 7 days."},
    {"question": "payment methods accepted", "answer": "We accept UPI, Card, and Net Banking."},
    {"question": "customer care contact number", "answer": "Customer care: 1800-XXX-XXXX"},
]

# TEXT CLEANING
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# FIND ANSWER
def get_answer(user_question):
    cleaned_q = clean_text(user_question)
    faq_questions = [faq['question'] for faq in faqs]
    all_texts = faq_questions + [cleaned_q]
    
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(all_texts)
    
    similarities = cosine_similarity(matrix[-1], matrix[:-1])[0]
    best_idx = np.argmax(similarities)
    
    if similarities[best_idx] < 0.2:
        return "Sorry, I couldn't find the information."
    
    return faqs[best_idx]['answer']

# CHAT LOOP
print("=" * 40)
print("🤖 Welcome to FAQ Chatbot!")
print("Type 'quit' to exit")
print("=" * 40)

while True:
    user_input = input("\nYou: ")
    
    if user_input.lower() == 'quit':
        print("Bot: Thank you! Goodbye 👋")
        break
    
    answer = get_answer(user_input)
    print(f"Bot: {answer}")