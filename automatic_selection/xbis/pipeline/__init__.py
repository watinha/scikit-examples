import re, codecs, np, random

from sklearn import tree, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold

class BibParser:
    def __init__ (self, write_files = True):
        self.texts_list = []
        self._write_files = write_files

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
                    sys.exit(1)

                for bib_index in range(len(titles)):
                    insert = 'selecionado' if inserir[bib_index][1] == 'true' else 'removido'
                    abstract = re.sub('[\n\r]', ' ', abstracts[bib_index][1])
                    content = u'%s\n%s' % (titles[bib_index][1], abstract)
                    if self._write_files:
                        newfile = codecs.open('corpus/%s/%s-%d.txt' %
                                (folder, insert, (bib_index + (file_index * 1000))), 'w', encoding='utf-8')
                        print('from %s writing file %s/%s-%d.txt' %
                                (filename, folder, insert, (bib_index + (file_index * 1000))))
                        newfile.write(content.decode())
                        newfile.close()
                    self.texts_list.append({
                        'content': content.decode(),
                        'category': insert
                    })
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
        result = {
            'features': features,
            'categories': np.array(categories)
        }
        print (result)
        return result

class DecisionTreeClassifier:
    def __init__ (self, seed):
        self._seed = seed

    def execute (self, dataset):
        print('===== Decision Tree Classifier =====')
        X = dataset['features']
        y = dataset['categories']
        random.seed(self._seed)
        model = tree.DecisionTreeClassifier(criterion='entropy', random_state=self._seed)
        kfold = StratifiedKFold(n_splits=5, random_state=self._seed)
        scores = cross_val_score(model, X, y, cv=kfold, scoring='f1_macro')
        print(scores)
        print("OUR APPROACH F-measure: %s on average and %s SD" % (scores.mean(), scores.std()))
        dataset['decision_tree_scores'] = scores
        return dataset

class SVMClassifier:
    def __init__ (self, seed):
        self._seed = seed

    def execute (self, dataset):
        print('===== SVM Classifier =====')
        X = dataset['features']
        y = dataset['categories']
        random.seed(self._seed)
        model = svm.SVC(gamma='scale')
        kfold = StratifiedKFold(n_splits=5, random_state=self._seed)
        scores = cross_val_score(model, X, y, cv=kfold, scoring='f1_macro')
        print(scores)
        print("OUR APPROACH F-measure: %s on average and %s SD" % (scores.mean(), scores.std()))
        dataset['svm_scores'] = scores
        return dataset
