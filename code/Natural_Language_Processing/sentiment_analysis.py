# Natural Language Processing

# Importing the libraries
import pandas as pd
import random

# Cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# Importing the dataset
dataset = pd.read_csv(
    'Restaurant_Reviews.tsv', delimiter='\t', quoting=3)


nltk.download('stopwords')

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(
        stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.20, random_state=0)

# Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
print("Accuracy Naive Bayes")
print(accuracy_score(y_test, y_pred))

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)


# Linear classifier
log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
print("Accuracy Linear Regresion")
print(accuracy_score(y_test, y_pred))


# show text review
j = random.randint(0, len(X_test)-7)
for i in range(j, j+7):
    print(y_pred[0])
    ind = X.tolist().index(X_test[i].tolist())
    print(corpus[ind].strip())
