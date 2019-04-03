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
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

model.fit(X_train, y_train, epochs=100,
          verbose=False, validation_data=(X_test, y_test),
          batch_size=10)

print('')
print('Keras with MLP with 10 hidden neurons')
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=500)
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

vocab_size = len(tokenizer.word_index) + 1

X_train = pad_sequences(X_train, padding='post', maxlen=100)
X_test = pad_sequences(X_test, padding='post', maxlen=100)

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size,
                           output_dim=50,
                           input_length=100))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

model.fit(X_train, y_train, epochs=100,
          verbose=False, validation_data=(X_test, y_test),
          batch_size=10)

print('')
print('Keras with MLP with 10 hidden neurons and Embeddings')
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

import numpy as np

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix

embedding_dim = 50
embedding_matrix = create_embedding_matrix(
     'data/glove.6B/glove.6B.50d.txt',
     tokenizer.word_index, embedding_dim)

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size,
                           output_dim=50,
                           weights=[embedding_matrix],
                           input_length=100,
                           trainable=True))
#model.add(layers.Conv1D(128, 5, activation='relu'))
#model.add(layers.GlobalMaxPool1D())
#model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

model.fit(X_train, y_train, epochs=100,
          verbose=False, validation_data=(X_test, y_test),
          batch_size=10)

print('')
print('Keras with MLP with 10 hidden neurons and pre-trained Embeddings')
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
