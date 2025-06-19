from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Example training data
training_data = [
    "This is spam",
    "This is not spam",
    "Hello world",
]

# Initialize and fit the TfidfVectorizer
tfidf = TfidfVectorizer()
tfidf.fit(training_data)

# Save the vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("Vectorizer saved successfully!")

