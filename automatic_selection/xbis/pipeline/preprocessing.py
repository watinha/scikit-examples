from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

class StopWordsFilter:
    def execute (self, texts_list):
        print '===== Removing stop words ====='
        result = []
        for text in texts_list:
            tokens = word_tokenize(text['content'])
            result.append({
                'content': (' ').join([
                    word for word in tokens
                    if not word in stopwords.words('english') ]),
                'category': text['category']
            })

        return result

class PorterStemmerFilter:
    def execute (self, texts_list):
        print '===== Stemming words ====='
        result = []
        stemmer = PorterStemmer()
        for text in texts_list:
            tokens = word_tokenize(text['content'])
            result.append({
                'content': (' ').join([
                    stemmer.stem(word) for word in tokens ]),
                'category': text['category']
            })

        return result
