import pandas as pd
from sklearn.model_selection import train_test_split

# Load the IMDB Movie Reviews dataset
data = pd.read_csv('data/raw_data/IMDB_Dataset.csv')

# Display the first few rows of the dataset
print(data.head())

# Split the dataset into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the training and test sets
train_data.to_csv('data/processed_data/train_data.csv', index=False)
test_data.to_csv('data/processed_data/test_data.csv', index=False)

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import pickle

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    # Remove punctuation and stop words
    tokens = [word for word in tokens if word not in string.punctuation and word not in stopwords.words('english')]
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Apply preprocessing to training and test sets
train_data['review'] = train_data['review'].apply(preprocess_text)
test_data['review'] = test_data['review'].apply(preprocess_text)

# Save preprocessed data
train_data.to_csv('data/processed_data/train_data_preprocessed.csv', index=False)
test_data.to_csv('data/processed_data/test_data_preprocessed.csv', index=False)

# Vectorize the preprocessed text
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_data['review']).toarray()
X_test = vectorizer.transform(test_data['review']).toarray()

# Save the vectorizer and vectorized data
with open('models/vectorizer.pickle', 'wb') as file:
    pickle.dump(vectorizer, file)
with open('data/processed_data/X_train.pickle', 'wb') as file:
    pickle.dump(X_train, file)
with open('data/processed_data/X_test.pickle', 'wb') as file:
    pickle.dump(X_test, file)

from imblearn.over_sampling import SMOTE

# Load preprocessed data
train_data = pd.read_csv('data/processed_data/train_data_preprocessed.csv')
test_data = pd.read_csv('data/processed_data/test_data_preprocessed.csv')

# Extract features and labels
X_train = train_data['review']
y_train = train_data['sentiment']
X_test = test_data['review']
y_test = test_data['sentiment']

# Vectorize the preprocessed text
X_train_vectorized = vectorizer.fit_transform(X_train).toarray()
X_test_vectorized = vectorizer.transform(X_test).toarray()

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_vectorized, y_train)

# Save the balanced data
with open('data/processed_data/X_train_balanced.pickle', 'wb') as file:
    pickle.dump(X_train_balanced, file)
with open('data/processed_data/y_train_balanced.pickle', 'wb') as file:
    pickle.dump(y_train_balanced, file)