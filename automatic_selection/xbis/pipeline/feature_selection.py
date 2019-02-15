from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV, VarianceThreshold

class RFECVFeatureSelection:
    def __init__ (self, estimator):
        self._rfecv = RFECV(estimator = estimator,
                cv=StratifiedKFold(5), scoring='f1_macro')

    def execute (self, dataset):
        print '===== Feature selection - RFECV ====='
        dataset['features'] = self._rfecv.fit_transform(dataset['features'].toarray(), dataset['categories'])
        print dataset['features'].shape
        return dataset


class VarianceThresholdFeatureSelection:
    def __init__ (self, threshold):
        self._variance_threshold = VarianceThreshold(threshold=threshold)

    def execute (self, dataset):
        print '===== Feature selection - Variance Threshold ====='
        dataset['features'] = self._variance_threshold.fit_transform(dataset['features'])
        print dataset['features'].shape
        return dataset
