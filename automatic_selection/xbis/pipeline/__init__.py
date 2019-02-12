import re, codecs, np

from sklearn.feature_extraction.text import TfidfVectorizer

class BibParser:
    def __init__ (self):
        self.texts_list = []

    def execute (self, files_list):
        print('===== Reading bib and transforming to text =====')
        for file_index in range(len(files_list)):
            filename = files_list[file_index];
            with codecs.open(filename, 'r', encoding='utf-8') as bib_file:
                bibfile = bib_file.read()
                titles = re.findall('(title)\s*=\s\{([^\}]*)\}', bibfile)
                abstracts = re.findall('(abstract)\s*=\s\{([^\}]*)\}', bibfile)
                inserir = re.findall('(inserir)\s*=\s\{([^\}]*)\}', bibfile)
                folder = filename.split('/')[1].split('-')[0]

                if (len(titles) != len(abstracts) or len(titles) != len(inserir)):
                    print 'Different number of titles, abstracts and inserir values...'
                    sys.exit()

                for bib_index in range(len(titles)):
                    insert = 'selecionado' if inserir[bib_index][1] == 'true' else 'removido'
                    newfile = codecs.open('corpus/%s/%s-%d.txt' %
                            (folder, insert, (bib_index + (file_index * 1000))), 'w', encoding='utf-8')
                    print('from %s writing file %s/%s-%d.txt' %
                            (filename, folder, insert, (bib_index + (file_index * 1000))))
                    abstract = re.sub('[\n\r]', ' ', abstracts[bib_index][1])
                    content = u'%s\n%s' % (titles[bib_index][1], abstract)
                    newfile.write(content.decode())
                    self.texts_list.append({
                        'content': content.decode(),
                        'category': insert
                    })
                    newfile.close()
                bib_file.close()

        return self.texts_list

class GenerateDataset:
    def execute (self, texts_list):
        print('===== Reading text and vectorizing =====')
        texts = [ text_data['content'] for text_data in texts_list ]
        categories = [ 1 if text_data['category'] == 'selecionado' else 0
                for text_data in texts_list ]
        vectorizer = TfidfVectorizer()
        features = vectorizer.fit_transform(texts)
        return {
            'features': features,
            'categories': np.array(categories)
        }
