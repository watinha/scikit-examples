import np

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


class EmbeddingsFeatureSelection:
    def __init__ (self, k=10000, random_state=42,
                  vectorizer=TfidfVectorizer(),
                  glove_file='glove.6B.200d.txt', embedding_dim=200):
        self._k = k
        self._random_state = 42
        self._vectorizer = vectorizer
        self._glove_file = glove_file
        self._embedding_dim = embedding_dim

    def execute (self, dataset):
        print('===== Feature selection - Glove Embeddings =====')
        texts = [ text_data['content'] for text_data in dataset ]
        self._vectorizer.fit(texts)

        if (len(self._vectorizer.vocabulary_) < self._k):
            print('Number of unique words is smaller than number of clusters (%d < %d)' %
                    (len(self._vectorizer.vocabulary_), self._k))
            return dataset

        print('===== Glove News Embeddings loading from %s =====' % (self._glove_file))
        embedding_dim = self._embedding_dim
        self._vocab_size = len(self._vectorizer.vocabulary_) + 1
        self._embedding_matrix = np.zeros((self._vocab_size, embedding_dim))
        with open(self._glove_file) as f:
            for line in f:
                word, *vector = line.split()
                if word in self._vectorizer.vocabulary_:
                    idx = self._vectorizer.vocabulary_[word]
                    self._embedding_matrix[idx] = np.array(
                        vector, dtype=np.float32)[:embedding_dim]

        print('===== K-Means %d =====' % (self._k))
        model = KMeans(n_clusters=self._k, random_state=self._random_state)
        model.fit(self._embedding_matrix)

        print('===== replacing similar words by similarity =====')
        for text_data in dataset:
            tokens = word_tokenize(text_data['content'])
            new_tokens = []
            for token in tokens:
                try:
                    word_embedding = self._embedding_matrix[self._vectorizer.vocabulary_[token]]
                    word_cluster = model.predict(np.array([word_embedding]))[0]
                    new_tokens.append(str(word_cluster))
                except:
                    #print('Key not found in index, removing word...')
                    pass

            text_data['content'] = ' '.join(new_tokens)

        return dataset

