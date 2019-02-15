import np, random

from sklearn import tree, metrics, svm, naive_bayes
from sklearn.model_selection import cross_val_score, StratifiedKFold

class SimpleClassifier:
    def __init__ (self, seed):
        self._seed = seed

    def execute (self, dataset):
        X = dataset['features']
        y = dataset['categories']
        random.seed(self._seed)
        kfold = StratifiedKFold(n_splits=5, random_state=self._seed)
        model = self.get_classifier()
        scores = cross_val_score(model, X, y, cv=kfold, scoring='f1_macro')
        print(scores)
        print("OUR APPROACH F-measure: %s on average and %s SD" % (scores.mean(), scores.std()))
        dataset['%s_scores' % self.classifier_name] = scores
        return dataset


class DecisionTreeClassifier (SimpleClassifier):
    def __init__ (self, seed):
        SimpleClassifier.__init__(self, seed)
        self.classifier_name = 'decision_tree'

    def get_classifier (self):
        print('===== Decision Tree Classifier =====')
        return tree.DecisionTreeClassifier(criterion='entropy', random_state=self._seed)


class SVMClassifier (SimpleClassifier):
    def __init__ (self, seed):
        SimpleClassifier.__init__(self, seed)
        self.classifier_name = 'svm'

    def get_classifier (self):
        print('===== SVM Classifier =====')
        return svm.SVC(gamma='scale', random_state=self._seed)


class LinearSVMClassifier (SimpleClassifier):
    def __init__ (self, seed):
        SimpleClassifier.__init__(self, seed)
        self.classifier_name = 'svm'

    def get_classifier (self):
        print('===== Linear SVM Classifier =====')
        return svm.LinearSVC(random_state=self._seed)


class NaiveBayesClassifier (SimpleClassifier):
    def __init__ (self, seed):
        SimpleClassifier.__init__(self, seed)
        self.classifier_name = 'naive_bayes'

    def get_classifier (self):
        print('===== NaiveBayes Classifier =====')
        return naive_bayes.GaussianNB()

    def execute (self, dataset):
        X = dataset['features']
        y = dataset['categories']
        random.seed(self._seed)
        kfold = StratifiedKFold(n_splits=5, random_state=self._seed)
        model = self.get_classifier()
        if (X.shape[1] > 1000):
            scores = cross_val_score(model, X.toarray(), y, cv=kfold, scoring='f1_macro')
        else:
            scores = cross_val_score(model, X, y, cv=kfold, scoring='f1_macro')

        print(scores)
        print("OUR APPROACH F-measure: %s on average and %s SD" % (scores.mean(), scores.std()))
        dataset['%s_scores' % self.classifier_name] = scores
        return dataset
