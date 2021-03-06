import re, codecs, np, sys, csv

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer

class BibParser:
    def __init__ (self, write_files = True, project_folder='', only_titles=False):
        self.texts_list = []
        self._write_files = write_files
        self._project_folder = project_folder
        self._only_titles = only_titles

    def execute (self, files_list):
        print('===== Reading bib and transforming to text =====')
        for file_index in range(len(files_list)):
            filename = files_list[file_index];
            with codecs.open(filename, 'r', encoding='utf-8') as bib_file:
                bibfile = bib_file.read()
                titles = re.findall('([^book][^short]title)\s*=\s\{([^\}]*)\}', bibfile)
                abstracts = re.findall('(abstract)\s*=\s\{([^\}]*)\}', bibfile)
                inserir = re.findall('(inserir)\s*=\s\{([^\}]*)\}', bibfile)
                folder = filename.split('/')[2].split('-')[0]

                if (len(titles) != len(abstracts) or len(titles) != len(inserir)):
                    print('Different number of titles, abstracts and inserir values...')
                    print('File: %s' % (filename))
                    print('Titles: %d\nAbstracts: %d\nInserir: %d' %
                            (len(titles), len(abstracts), len(inserir)))
                    sys.exit(1)

                for bib_index in range(len(titles)):
                    insert = 'selecionado' if inserir[bib_index][1] == 'true' else 'removido'
                    title = re.sub('[\n\r]', ' ', titles[bib_index][1])
                    abstract = re.sub('[\n\r]', ' ', abstracts[bib_index][1])
                    content = u'%s\n%s' % (title, abstract)
                    if self._write_files:
                        newfile = codecs.open('corpus/%s/%s/%s-%d.txt' %
                                (self._project_folder, folder, insert,
                                (bib_index + (file_index * 1000))), 'w', encoding='utf-8')
                        print('from %s writing file corpus/%s/%s/%s-%d.txt' %
                                (filename, self._project_folder, folder,
                                 insert, (bib_index + (file_index * 1000))))
                        newfile.write(content)
                        newfile.close()
                    if self._only_titles:
                        content = content.split('\n')[0]
                    self.texts_list.append({
                        'content': content,
                        'category': insert
                    })
                bib_file.close()

        return self.texts_list


class GenerateDataset:
    def __init__ (self, vectorizer=TfidfVectorizer()):
        self._vectorizer = vectorizer

    def execute (self, texts_list):
        print('===== Reading text and vectorizing =====')
        texts = [ text_data['content'] for text_data in texts_list ]
        categories = [ 1 if text_data['category'] == 'selecionado' else 0
                for text_data in texts_list ]
        features = self._vectorizer.fit_transform(texts)
        result = {
            'texts': texts,
            'features': features,
            'categories': np.array(categories)
        }
        print (result['features'].shape)
        return result


class GenerateSequences:
    def __init__ (self, num_words=500, maxlen=500):
        self._num_words = num_words
        self._maxlen = maxlen

    def execute (self, result):
        print('===== Transforming texts to sequence representation =====')
        tokenizer = Tokenizer(num_words=self._num_words)
        tokenizer.fit_on_texts(result['texts'])
        features = tokenizer.texts_to_sequences(result['texts'])
        features = pad_sequences(features, padding='post', maxlen=self._maxlen)
        result['word_index'] = tokenizer.word_index
        result['features'] = features
        return result
