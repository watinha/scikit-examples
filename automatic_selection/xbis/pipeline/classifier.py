import np, random

from sklearn import tree, metrics, svm
from sklearn.model_selection import cross_val_score, StratifiedKFold

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
