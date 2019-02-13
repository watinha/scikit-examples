from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

class TextFilterComposite:
    def __init__ (self, filters):
        self._filters = filters

    def _filter (self, tokens):
        result = tokens
        for f in self._filters:
            result = f.filter(result)
        return (' ').join(result)

    def execute (self, texts_list):
        print '===== Executing Text Filter ====='
        result = []
        for text in texts_list:
            tokens = word_tokenize(text['content'])
            filtered_text = self._filter(tokens)
            result.append({
                'content': filtered_text,
                'category': text['category']
            })

        return result

class StopWordsFilter:
    def __init__ (self):
        print '===== Removing stop words ====='

    def filter (self, tokens):
        return [ word for word in tokens
                 if not word in stopwords.words('english') ]

class PorterStemmerFilter:
    def __init__ (self):
        print '===== Stemming words ====='
        self._stemmer = PorterStemmer()

    def filter (self, tokens):
        return [ self._stemmer.stem(word) for word in tokens ]
