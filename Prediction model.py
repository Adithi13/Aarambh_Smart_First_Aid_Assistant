"""

this code is for checking the model accuracy


"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


df = pd.read_csv("Health.csv")
df['Symptoms'] = df['Symptoms'].str.lower()  


disease_mapping = {
    'HeartDisease': 0,
    'JointDislocation': 1,
    'LowBp': 2,
    'SnakeBite': 3
}
df['disease_label'] = df['Disease'].map(disease_mapping)


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha()]
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df['preprocessed_Symptoms'] = df['Symptoms'].apply(preprocess_text)


X_train, X_test, y_train, y_test = train_test_split(df['preprocessed_Symptoms'], df['disease_label'], test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)


y_pred = clf.predict(X_test_tfidf)


print(classification_report(y_test, y_pred, target_names=disease_mapping.keys()))
