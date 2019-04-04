import np, random

from sklearn import tree, metrics, svm, naive_bayes, ensemble, linear_model, neural_network
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV, train_test_split

class SimpleClassifier:
    def __init__ (self, seed):
        self._seed = seed

    def execute (self, dataset):
        X = dataset['features']
        y = dataset['categories']
        random.seed(self._seed)
        kfold = StratifiedKFold(n_splits=5, random_state=self._seed)
        model = self.get_classifier(X, y)
        scores = cross_validate(model, X, y, cv=kfold, scoring=['f1_macro', 'precision_macro', 'recall_macro'])
        print("OUR APPROACH F-measure: %s on average and %s SD" %
                (scores['test_f1_macro'].mean(), scores['test_f1_macro'].std()))
        print("OUR APPROACH Precision: %s on average and %s SD" %
                (scores['test_precision_macro'].mean(), scores['test_precision_macro'].std()))
        print("OUR APPROACH Recall: %s on average and %s SD" %
                (scores['test_recall_macro'].mean(), scores['test_recall_macro'].std()))

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)
        scores['probabilities'] = probabilities[:, 1]
        scores['y_test'] = y_test

        dataset['%s_scores' % self.classifier_name] = scores
        return dataset

class RandomForestClassifier (SimpleClassifier):
    def __init__ (self, seed=42, criterion='entropy'):
        SimpleClassifier.__init__(self, seed)
        self.classifier_name = 'random_forest'
        self._criterion = criterion

    def get_classifier (self, X, y):
        print('===== Random Forest Classifier =====')
        print('===== Hyperparameter tunning  =====')
        model = ensemble.RandomForestClassifier(random_state=self._seed)
        params = {
            'n_estimators': [5, 10, 100],
            'criterion': ["gini", "entropy"],
            'max_depth': [10, 50, 100, None],
            'min_samples_split': [2, 10, 100],
            'class_weight': [None, 'balanced']
        }
        cfl = GridSearchCV(model, params, cv=5)
        cfl.fit(X, y)
        for param, value in cfl.best_params_.items():
            print("%s : %s" % (param, value))

        model = ensemble.RandomForestClassifier(random_state=self._seed)
        model.set_params(**cfl.best_params_)
        return model

class DecisionTreeClassifier (SimpleClassifier):
    def __init__ (self, seed=42, criterion='entropy'):
        SimpleClassifier.__init__(self, seed)
        self.classifier_name = 'decision_tree'
        self._criterion = criterion

    def get_classifier (self, X, y):
        print('===== Decision Tree Classifier =====')
        print('===== Hyperparameter tunning  =====')
        model = tree.DecisionTreeClassifier()
        params = {
            'criterion': ["gini", "entropy"],
            'max_depth': [10, 50, 100, None],
            'min_samples_split': [2, 10, 100],
            'class_weight': [None, 'balanced']
        }
        cfl = GridSearchCV(model, params, cv=5)
        cfl.fit(X, y)
        for param, value in cfl.best_params_.items():
            print("%s : %s" % (param, value))

        model = tree.DecisionTreeClassifier(random_state=self._seed)
        model.set_params(**cfl.best_params_)
        return model


class LogisticRegressionClassifier (SimpleClassifier):
    def __init__ (self, seed):
        SimpleClassifier.__init__(self, seed)
        self.classifier_name = 'LR'

    def get_classifier (self, X, y):
        print('===== LR Classifier =====')
        return linear_model.LogisticRegression(random_state=self._seed)


class MLPClassifier (SimpleClassifier):
    def __init__ (self, seed):
        SimpleClassifier.__init__(self, seed)
        self.classifier_name = 'MLP'

    def get_classifier (self, X, y):
        print('===== MLP Classifier =====')
        return neural_network.MLPClassifier(random_state=self._seed)


class SVMClassifier (SimpleClassifier):
    def __init__ (self, seed):
        SimpleClassifier.__init__(self, seed)
        self.classifier_name = 'svm'

    def get_classifier (self, X, y):
        print('===== SVM Classifier =====')
        return svm.SVC(gamma='scale', random_state=self._seed)


class LinearSVMClassifier (SimpleClassifier):
    def __init__ (self, seed):
        SimpleClassifier.__init__(self, seed)
        self.classifier_name = 'svm'

    def get_classifier (self, X, y):
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
