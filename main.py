#importing the require modules
import pandas as pd
import numpy as np

#loading dataset
df = pd.read_csv("datasets/spam.csv", encoding="ISO-8859-1", header = 0, names = ['type', 'message'])

print(df.head())


#sklearn
from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

#build the model
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

text = text.ENGLISH_STOP_WORDS

#NLTK
import nltk 
nltk.download("stopwords")
nltk.download('punkt')

from nltk.stem.porter import *
from nltk.corpus import stopwords
stop = stopwords.words('english')

#tokenize
df['tokens'] = df.apply(lambda x: nltk.word_tokenize(str(x['message'])), axis = 1)

#remove stopwords
df['tokens'] = df['tokens'].apply(lambda x: [item for item in x if item not in stop])

#apply porter stemming 
stemmer = PorterStemmer()
df['tokens'] = df['tokens'].apply(lambda x: [stemmer.stem(item) for item in x])

#Unifying the string again
df['tokens'] = df['tokens'].apply(lambda x: ' '.join(x))

np.isnan(df.values.any())

#making split
x_train, x_test, y_train, y_test = train_test_split(
    df['tokens'], 
    df['type'], 
    test_size= 0.2
    )


# Create vectorizer
vectorizer = CountVectorizer(
    strip_accents = 'ascii', 
    lowercase = True
    )

# Fit vectorizer & transform it
vectorizer_fit = vectorizer.fit(x_train)
x_train_transformed = vectorizer_fit.transform(x_train)
x_test_transformed = vectorizer_fit.transform(x_test)


#training the model
naive_bayes = MultinomialNB()
naive_bayes_fit = naive_bayes.fit(x_train_transformed, y_train)

# Making predictions
train_predict = naive_bayes_fit.predict(x_train_transformed)
test_predict = naive_bayes_fit.predict(x_test_transformed)

def get_scores(y_real, predict):
  ba_train = balanced_accuracy_score(y_real, predict)
  cm_train = confusion_matrix(y_real, predict)

  return ba_train, cm_train 

def print_scores(scores):
  return f"Balanced Accuracy: {scores[0]}\nConfussion Matrix:\n {scores[1]}"

train_scores = get_scores(y_train, train_predict)
test_scores = get_scores(y_test, test_predict)


print("## Train Scores")
print(print_scores(train_scores))
print("\n\n## Test Scores")
print(print_scores(test_scores))