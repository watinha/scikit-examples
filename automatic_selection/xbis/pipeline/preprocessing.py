from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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

