import csv, re

class CSVReporter:
    def __init__ (self, filename):
        self._filename = filename
        self._precision = {}
        self._recall = {}
        self._fscore = {}
        csv_file = open(self._filename, 'w', newline='')
        self._csv_writer = csv.writer(csv_file, delimiter=';')
        self._classifiers = []

    def execute (self, dataset):
        keys = list(dataset.keys())
        classifiers = [ classifier_score
                for classifier_score in keys
                if re.match('.*_scores$', classifier_score) != None ]
        self._classifiers = [ re.findall('(.*)_scores$', classifier_score)[0]
                                for classifier_score in classifiers ]
        if (len(self._precision) == 0):
            self._csv_writer.writerow(['fold'] + (self._classifiers * 3))
        for classifier_score in classifiers:
            scores = dataset[classifier_score]
            classifier_name = re.findall('(.*)_scores$', classifier_score)[0]
            precision = scores['test_precision_macro'].tolist()
            recall = scores['test_recall_macro'].tolist()
            fscore = scores['test_f1_macro'].tolist()
            if (self._precision.get(classifier_name) == None):
                self._precision[classifier_name] = []
            self._precision[classifier_name] = self._precision[classifier_name] + precision
            if (self._recall.get(classifier_name) == None):
                self._recall[classifier_name] = []
            self._recall[classifier_name] = self._recall[classifier_name] + recall
            if (self._fscore.get(classifier_name) == None):
                self._fscore[classifier_name] = []
            self._fscore[classifier_name] = self._fscore[classifier_name] + fscore

    def report (self):
        for i in range(0, len(self._precision[self._classifiers[0]])):
            precision_values = []
            recall_values = []
            fscore_values = []
            for j in self._classifiers:
                precision_values += [self._precision[j][i]]
                recall_values += [self._recall[j][i]]
                fscore_values += [self._fscore[j][i]]

            row = ['Fold-%d' % i] + precision_values + recall_values + fscore_values
            self._csv_writer.writerow(row)

