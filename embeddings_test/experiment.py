import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/imdb_labelled.txt',
                 names=['sentence', 'label'],
                 sep='\t')

sentences = df['sentence'].values
y = df['label'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.20, random_state=42)

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
print('Logistic Regression')
print(lr_model.score(X_test, y_test))


from keras.models import Sequential
from keras import layers

input_dim = X_train.shape[1]

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile()
model.summary()

model.fit(X_train, y_train, epoch=100,
          verbose=False, validation_data=(X_test, y_test),
          batch_size=10)

print('')
print('Keras with MLP with 10 hidden neurons')
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
