# import streamlit as st
# import pickle
# import string
# from nltk.corpus import stopwords
# import nltk
# from nltk.stem.porter import PorterStemmer

# ps = PorterStemmer()
# nltk.download('punkt_tab')


# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         y.append(ps.stem(i))

#     return " ".join(y)

# tfidf = pickle.load(open('C:/Users/ahmed/OneDrive/Desktop/EMAIL-SPAM-CLASSIFIER-AI-MINI-PROJECT-main/EMAIL-SPAM-CLASSIFIER-AI-MINI-PROJECT-main/vectorizer.pkl', 'rb'))
# model = pickle.load(open('C:/Users/ahmed/OneDrive/Desktop/EMAIL-SPAM-CLASSIFIER-AI-MINI-PROJECT-main/EMAIL-SPAM-CLASSIFIER-AI-MINI-PROJECT-main/model.pkl','rb'))

# st.title("Email/SMS Spam Classifier")

# input_sms = st.text_area("Enter the message")

# if st.button('Predict'):

#     # 1. preprocess
#     transformed_sms = transform_text(input_sms)
#     # 2. vectorize
#     vector_input = tfidf.transform([transformed_sms])
#     # 3. predict
#     result = model.predict(vector_input)[0]
#     # 4. Display
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")

# import streamlit as st
# import pickle
# import string
# from nltk.corpus import stopwords
# import nltk
# from nltk.stem.porter import PorterStemmer

# # Download necessary resources
# nltk.download('punkt')
# nltk.download('stopwords')

# ps = PorterStemmer()

# def transform_text(text):
#     text = text.lower()  # Lowercase the text
#     text = nltk.word_tokenize(text)  # Tokenize the text

#     # Remove non-alphanumeric tokens
#     text = [i for i in text if i.isalnum()]

#     # Remove stopwords and punctuation
#     text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

#     # Perform stemming
#     text = [ps.stem(i) for i in text]

#     return " ".join(text)

# # Load pre-trained models and vectorizer
# vectorizer_path = 'C:/Users/ahmed/OneDrive/Desktop/EMAIL-SPAM-CLASSIFIER-AI-MINI-PROJECT-main/EMAIL-SPAM-CLASSIFIER-AI-MINI-PROJECT-main/vectorizer.pkl'
# model_path = 'C:/Users/ahmed/OneDrive/Desktop/EMAIL-SPAM-CLASSIFIER-AI-MINI-PROJECT-main/EMAIL-SPAM-CLASSIFIER-AI-MINI-PROJECT-main/model.pkl'


# # Streamlit interface
# st.title("Email/SMS Spam Classifier")
# input_sms = st.text_area("Enter the message")

# if st.button('Predict'):
#     # Preprocess the input text
#     transformed_sms = transform_text(input_sms)
    
#     # Vectorize the input text
#     vector_input = tfidf.transform([transformed_sms])
    
#     # Make prediction
#     result = model.predict(vector_input)[0]
    
#     # Display the result
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")






# import streamlit as st
# import pickle
# import string
# from nltk.corpus import stopwords
# import nltk
# from nltk.stem.porter import PorterStemmer

# # Download necessary NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')

# ps = PorterStemmer()

# def transform_text(text):
#     text = text.lower()  # Lowercase the text
#     text = nltk.word_tokenize(text)  # Tokenize the text

#     # Remove non-alphanumeric tokens
#     text = [i for i in text if i.isalnum()]

#     # Remove stopwords and punctuation
#     text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

#     # Perform stemming
#     text = [ps.stem(i) for i in text]

#     return " ".join(text)

# # Load the vectorizer and model
# try:
#     with open('vectorizer.pkl', 'rb') as f:
#         tfidf = pickle.load(f)
#     with open('model.pkl', 'rb') as f:
#         model = pickle.load(f)
# except FileNotFoundError:
#     st.error("Model or vectorizer file not found. Please check file paths.")
#     st.stop()

# # Streamlit interface
# st.title("Email/SMS Spam Classifier")
# input_sms = st.text_area("Enter the message")

# if st.button('Predict'):
#     if not input_sms.strip():
#         st.warning("Please enter a valid message.")
#     else:
#         # Preprocess the input text
#         transformed_sms = transform_text(input_sms)

#         try:
#             # Vectorize the input text
#             vector_input = tfidf.transform([transformed_sms])

#             # Make prediction
#             result = model.predict(vector_input)[0]

#             # Display the result
#             if result == 1:
#                 st.header("Spam")
#             else:
#                 st.header("Not Spam")
#         except Exception as e:
#             st.error(f"Error during prediction: {e}")







# import streamlit as st
# import pickle
# import string
# from nltk.corpus import stopwords
# import nltk
# from nltk.stem.porter import PorterStemmer

# ps = PorterStemmer()
# nltk.download('punkt')
# nltk.download('stopwords')

# def transform_text(text):
#     text = text.lower()  # Lowercase the text
#     text = nltk.word_tokenize(text)  # Tokenize the text

#     # Remove non-alphanumeric tokens
#     text = [i for i in text if i.isalnum()]

#     # Remove stopwords and punctuation
#     text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

#     # Perform stemming
#     text = [ps.stem(i) for i in text]

#     return " ".join(text)

# # Load the model
# model = pickle.load(open('C:/Users/ahmed/OneDrive/Desktop/EMAIL-SPAM-CLASSIFIER-AI-MINI-PROJECT-main/EMAIL-SPAM-CLASSIFIER-AI-MINI-PROJECT-main/model.pkl','rb'))

# # Streamlit interface
# st.title("Email/SMS Spam Classifier")
# input_sms = st.text_area("Enter the message")

# if st.button('Predict'):
#     # Preprocess the input text
#     transformed_sms = transform_text(input_sms)
    
#     # Vectorize the input text (fit on input during prediction)
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     tfidf = TfidfVectorizer()  # You can replace this with your trained vectorizer if available
#     vector_input = tfidf.fit_transform([transformed_sms])  # This will fit and transform during prediction
    
#     # Make prediction
#     result = model.predict(vector_input)[0]
    
#     # Display the result
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")



# import streamlit as st
# import pickle
# import string
# from nltk.corpus import stopwords
# import nltk
# from nltk.stem.porter import PorterStemmer

# # Download necessary resources
# nltk.download('punkt')
# nltk.download('stopwords')

# ps = PorterStemmer()

# # Text preprocessing function
# def transform_text(text):
#     text = text.lower()  # Lowercase the text
#     text = nltk.word_tokenize(text)  # Tokenize the text

#     # Remove non-alphanumeric tokens
#     text = [i for i in text if i.isalnum()]

#     # Remove stopwords and punctuation
#     text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

#     # Perform stemming
#     text = [ps.stem(i) for i in text]

#     return " ".join(text)

# # Load the pre-trained vectorizer
# vectorizer_path = 'C:/Users/ahmed/OneDrive/Desktop/EMAIL-SPAM-CLASSIFIER-AI-MINI-PROJECT-main/EMAIL-SPAM-CLASSIFIER-AI-MINI-PROJECT-main/vectorizer.pkl'
# model_path = 'C:/Users/ahmed/OneDrive/Desktop/EMAIL-SPAM-CLASSIFIER-AI-MINI-PROJECT-main/EMAIL-SPAM-CLASSIFIER-AI-MINI-PROJECT-main/model.pkl'

# try:
#     with open(vectorizer_path, 'rb') as f:
#         tfidf = pickle.load(f)  # Load the TfidfVectorizer
#     with open(model_path, 'rb') as f:
#         model = pickle.load(f)  # Load the pre-trained model
# except FileNotFoundError:
#     st.error("Required files (vectorizer.pkl and/or model.pkl) not found. Please ensure they are in the correct location.")
#     st.stop()

# # Streamlit interface
# st.title("Email/SMS Spam Classifier")
# input_sms = st.text_area("Enter the message")

# if st.button('Predict'):
#     # Preprocess the input text
#     transformed_sms = transform_text(input_sms)
    
#     # Vectorize the input text
#     vector_input = tfidf.transform([transformed_sms])
    
#     # Make prediction
#     result = model.predict(vector_input)[0]
    
#     # Display the result
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")




import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()  # Lowercase the text
    text = nltk.word_tokenize(text)  # Tokenize the text

    # Remove non-alphanumeric tokens
    text = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

    # Perform stemming
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# Load the pre-trained vectorizer and model
try:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)  # Load the TfidfVectorizer
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)  # Load the pre-trained model
except FileNotFoundError:
    st.error("Required files (vectorizer.pkl and/or model.pkl) not found. Please ensure they are in the correct location.")
    st.stop()

# Streamlit interface
st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Preprocess the input text
    transformed_sms = transform_text(input_sms)
    
    # Vectorize the input text
    vector_input = tfidf.transform([transformed_sms])
    
    # Make prediction
    result = model.predict(vector_input)[0]
    
    # Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
