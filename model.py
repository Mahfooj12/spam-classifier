# from sklearn.naive_bayes import MultinomialNB
# import pickle

# # Example training labels (1 = spam, 0 = not spam)
# training_labels = [1, 0, 0]

# # Transform the training data using the vectorizer
# training_vectors = tfidf.transform(training_data)

# # Train a simple model (e.g., Naive Bayes)
# model = MultinomialNB()
# model.fit(training_vectors, training_labels)

# # Save the model
# with open('model.pkl', 'wb') as f:
#     pickle.dump(model, f)

# print("Model saved successfully!")




# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# import pickle

# # Example training data
# training_data = [
#     "This is spam",
#     "Congratulations",
#     "This is a spam message.",
#     "This is not spam",
#     "Hello world",
# ]

# # Example training labels (1 = spam, 0 = not spam)
# training_labels = [1, 0, 0]

# # Load or define the TfidfVectorizer
# try:
#     # Load the pre-saved vectorizer if it exists
#     with open('vectorizer.pkl', 'rb') as f:
#         tfidf = pickle.load(f)
# except FileNotFoundError:
#     # If vectorizer.pkl does not exist, create and save a new one
#     tfidf = TfidfVectorizer()
#     tfidf.fit(training_data)
#     with open('vectorizer.pkl', 'wb') as f:
#         pickle.dump(tfidf, f)
#     print("Vectorizer saved successfully!")

# # Transform the training data using the vectorizer
# training_vectors = tfidf.transform(training_data)

# # Train a simple model (e.g., Naive Bayes)
# model = MultinomialNB()
# model.fit(training_vectors, training_labels)

# # Save the model
# with open('model.pkl', 'wb') as f:
#     pickle.dump(model, f)

# print("Model saved successfully!")



# Training script (train_model.py)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Example training data and labels
training_data = [
    "This is spam",
    "Congratulations",
    "This is a spam message.",
    "This is not spam",
    "Hello world",
]
training_labels = [1, 0, 1, 0, 0]

# Create and fit the TfidfVectorizer
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(training_data)

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, training_labels)

# Save the trained vectorizer and model
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model and vectorizer saved successfully!")


