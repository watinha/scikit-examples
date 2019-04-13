import np, random

from keras import layers
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import metrics
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV, train_test_split


class EmbeddingClassifier:
    def __init__ (self, seed):
        self._seed = seed

    def execute (self, dataset):
        X = dataset['features']
        y = dataset['categories']
        random.seed(self._seed)
        kfold = StratifiedKFold(n_splits=5, random_state=self._seed)
        model = self.get_classifier(X, y, dataset['word_index'])
        scores = cross_validate(model, X, y, cv=kfold, scoring=['f1_macro', 'precision_macro', 'recall_macro'])
        print("OUR APPROACH F-measure: %s on average and %s SD" %
                (scores['test_f1_macro'].mean(), scores['test_f1_macro'].std()))
        print("OUR APPROACH Precision: %s on average and %s SD" %
                (scores['test_precision_macro'].mean(), scores['test_precision_macro'].std()))
        print("OUR APPROACH Recall: %s on average and %s SD" %
                (scores['test_recall_macro'].mean(), scores['test_recall_macro'].std()))

        #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        #model.fit(X_train, y_train)
        #probabilities = model.predict_proba(X_test)
        #scores['probabilities'] = probabilities[:, 1]
        #scores['y_test'] = y_test

        dataset['%s_scores' % self.classifier_name] = scores
        return dataset

class MLPKerasGloveEmbeddingClassifier (EmbeddingClassifier):
    def __init__ (self, seed=42, activation='relu', neurons_number=10, embedding_dim=200, maxlen=500, glove_file='glove.6B.200d.txt'):
        EmbeddingClassifier.__init__(self, seed)
        self.classifier_name = 'MLPKerasGLOVEEmbedding'
        self._activation = activation
        self._seed = seed
        self._neurons_number = neurons_number
        self._glove_file = glove_file
        self._embedding_dim = embedding_dim
        self._maxlen = maxlen
        self._embedding_matrix = None


    def get_classifier (self, X, y, word_index):
        print('===== MLP Keras with %d hidden neuros and Glove Embedding =====' % (self._neurons_number))
        # generate embedding matrix
        if (self._embedding_matrix == None):
            embedding_dim = self._embedding_dim
            vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
            self._embedding_matrix = np.zeros((vocab_size, embedding_dim))
            with open(self._glove_file) as f:
                for line in f:
                    word, *vector = line.split()
                    if word in word_index:
                        idx = word_index[word]
                        self._embedding_matrix[idx] = np.array(
                            vector, dtype=np.float32)[:embedding_dim]

        def create_model ():
            input_dim = X.shape[1]
            model = Sequential()
            model.add(layers.Embedding(input_dim=vocab_size,
                                       output_dim=embedding_dim,
                                       weights=[self._embedding_matrix],
                                       input_length=self._maxlen,
                                       trainable=True))
            model.add(layers.GlobalMaxPool1D())
            model.add(layers.Dense(self._neurons_number, activation='relu'))
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
            model.summary()
            return model
        return KerasClassifier(build_fn=create_model, epochs=150, verbose=0)