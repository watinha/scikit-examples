import np, random

from sklearn import tree, metrics, svm, naive_bayes, ensemble
from sklearn.model_selection import cross_validate, StratifiedKFold

class SimpleClassifier:
    def __init__ (self, seed):
        self._seed = seed

    def execute (self, dataset):
        X = dataset['features']
        y = dataset['categories']
        random.seed(self._seed)
        kfold = StratifiedKFold(n_splits=5, random_state=self._seed)
        model = self.get_classifier()
        scores = cross_validate(model, X, y, cv=kfold, scoring=['f1_macro', 'precision_macro', 'recall_macro'])
        print("OUR APPROACH F-measure: %s on average and %s SD" %
                (scores['test_f1_macro'].mean(), scores['test_f1_macro'].std()))
        print("OUR APPROACH Precision: %s on average and %s SD" %
                (scores['test_precision_macro'].mean(), scores['test_precision_macro'].std()))
        print("OUR APPROACH Recall: %s on average and %s SD" %
                (scores['test_recall_macro'].mean(), scores['test_recall_macro'].std()))
        dataset['%s_scores' % self.classifier_name] = scores
        return dataset

class RandomForestClassifier (SimpleClassifier):
    def __init__ (self, seed=42, criterion='entropy'):
        SimpleClassifier.__init__(self, seed)
        self.classifier_name = 'random_forest'
        self._criterion = criterion

    def get_classifier (self):
        print('===== Random Forest Classifier =====')
        return ensemble.RandomForestClassifier(
                n_estimators=4, criterion=self._criterion, random_state=self._seed)


class DecisionTreeClassifier (SimpleClassifier):
    def __init__ (self, seed=42, criterion='entropy'):
        SimpleClassifier.__init__(self, seed)
        self.classifier_name = 'decision_tree'
        self._criterion = criterion

    def get_classifier (self):
        print('===== Decision Tree Classifier =====')
        return tree.DecisionTreeClassifier(criterion=self._criterion, random_state=self._seed)


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
        if (X.shape[1] > 3000):
            scores = cross_validate(model, X.toarray(), y, cv=kfold, scoring=['f1_macro', 'precision_macro', 'recall_macro'])
        else:
            scores = cross_validate(model, X, y, cv=kfold, scoring=['f1_macro', 'precision_macro', 'recall_macro'])

        print("OUR APPROACH F-measure: %s on average and %s SD" %
                (scores['test_f1_macro'].mean(), scores['test_f1_macro'].std()))
        print("OUR APPROACH Precision: %s on average and %s SD" %
                (scores['test_precision_macro'].mean(), scores['test_precision_macro'].std()))
        print("OUR APPROACH Recall: %s on average and %s SD" %
                (scores['test_recall_macro'].mean(), scores['test_recall_macro'].std()))
        dataset['%s_scores' % self.classifier_name] = scores
        return dataset
