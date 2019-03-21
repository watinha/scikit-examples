import arff, np, random
from sklearn import tree, metrics, svm
from sklearn.model_selection import cross_val_score, GroupKFold, GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

features = [
'childsNumber', 'textLength', # all features
'baseDPI', 'targetDPI', 'baseX',
'targetX', 'baseY', 'targetY', 'baseHeight', 'targetHeight', 'baseWidth',
'targetWidth', 'baseParentX', 'targetParentX', 'baseParentY', 'targetParentY', 'imageDiff',
'chiSquared', 'baseDeviceWidth', 'targetDeviceWidth', 'baseViewportWidth', 'targetViewportWidth',
'phash', 'crosscheck.base.a', 'crosscheck.target.a', 'crosscheck.SDR',
'crosscheck.disp', 'crosscheck.area', 'crosscheck.LDTD', 'baseLeft', 'targetLeft', 'baseRight',
'targetRight', 'baseParentLeft', 'targetParentLeft', 'baseParentRight', 'targetParentRight',
'diff.viewport', 'size', 'diff.left.relation', 'diff.right.relation', 'diff.parent.left.relation',
'diff.parent.right.relation', 'diff.left.viewport', 'diff.right.viewport', 'alignment',
'diff.parent.left.viewport', 'diff.parent.right.viewport', 'parent.alignment', 'diff.parent.y',
'diff.parent.y.height', 'diff.height.height', 'diff.width.viewport', 'imageDiff.size',
'chiSquared.size', 'imageDiff.size.viewport', 'chiSquared.size.viewport', 'pHashDistance.viewport',
    'size',  # features used in the article
    'diff.height.height',
    'diff.width.viewport',
    'diff.left.viewport',
    'diff.right.viewport',
    'diff.parent.y',
    'imageDiff.size',
    'chiSquared.size',
    'phash',
    'childsNumber', 'textLength'
]

crosscheck_features = [
    'crosscheck.SDR', 'crosscheck.disp', 'crosscheck.LDTD',
    'crosscheck.area', 'chiSquared'
]

arff_file = arff.load(open('xbis.arff', 'r'))
headers = [header[0] for header in arff_file['attributes']]
dataset = np.array(arff_file['data'])
features_index = [headers.index(feature) for feature in features]
crosscheck_index = map(lambda feature: headers.index(feature), crosscheck_features)
class_index = headers.index('Result')
url_index = headers.index('URL')

def column (column_name):
    return headers.index(column_name)

base_left = np.array(dataset[:,column('baseLeft')], dtype=float)
base_right = np.array(dataset[:,column('baseRight')], dtype=float)
target_left = np.array(dataset[:,column('targetLeft')], dtype=float)
target_right = np.array(dataset[:,column('targetRight')], dtype=float)
base_viewport = np.array(dataset[:,column('baseViewportWidth')], dtype=float)
target_viewport = np.array(dataset[:,column('targetViewportWidth')], dtype=float)
diff_out_viewport_left = (abs((base_right - base_viewport) - (target_right - target_viewport))).reshape(1, dataset.shape[0])
diff_out_viewport_right = (abs((base_left - base_viewport) - (target_left - target_viewport))).reshape(1, dataset.shape[0])

X = np.array([ dataset[:, index] for index in features_index ], dtype=float).T
X = np.concatenate((X, diff_out_viewport_left.T, diff_out_viewport_right.T), axis=1)
X_cross = np.array([ dataset[:, index] for index in crosscheck_index ], dtype=float).T
y = np.array(dataset[:, class_index], dtype=int)
urls = dataset[:, url_index]

random.seed(42)
model = tree.DecisionTreeClassifier(criterion='entropy', random_state=42)
X_new = X
rfecv = RFECV(model, cv=GroupKFold(n_splits=5), scoring='f1_macro')
rfecv.fit(X, y, groups=urls)
X_new = rfecv.transform(X)
print (X.shape)
print (X_new.shape)
headers.append('diff_out_viewport_left')
headers.append('diff_out_viewport_right')
features_index.append(headers.index('diff_out_viewport_left'))
features_index.append(headers.index('diff_out_viewport_right'))
print ([ headers[features_index[i]]
            for i in range(0, len(rfecv.ranking_))
            if rfecv.ranking_[i] == 1 ])

params = {
    'criterion': ["gini", "entropy"],
    'max_depth': [10, 30, 60, 100, None],
    'min_samples_split': [2, 10, 15, 30, 50, 100, 200, 1000]
}
cfl = GridSearchCV(model, params, cv=5)
cfl.fit(X_new, y)
for param, value in cfl.best_params_.items():
    print("%s : %s" % (param, value))
model = tree.DecisionTreeClassifier(random_state=42)
model.set_params(**cfl.best_params_)

random.seed(42)
folds = GroupKFold(n_splits=10)
scores = cross_val_score(model, X_new, y, cv=folds, groups=urls, scoring='f1_macro')
#print(scores)
print("OUR APPROACH F-measure: %s on average and %s SD" % (scores.mean(), scores.std()))
random.seed(42)
model = tree.DecisionTreeClassifier(criterion='entropy', random_state=42)
folds = GroupKFold(n_splits=10)
scores = cross_val_score(model, X_cross, y, cv=folds, groups=urls, scoring='f1_macro')
#print(scores)
print("CROSSCHECK   F-measure: %s on average and %s SD" % (scores.mean(), scores.std()))

#print('===== Feature selection =====')
##model = tree.DecisionTreeClassifier(criterion='entropy')
#model = svm.LinearSVC()
#rfecv = RFECV(model, cv=StratifiedKFold(5), scoring='f1_macro')
#X_new = rfecv.fit_transform(X, y, groups=urls)
#print (X.shape)
#print (X_new.shape)
#print (features_index)
#print (rfecv.ranking_)
#random.seed(42)
##model = tree.DecisionTreeClassifier(criterion='entropy', random_state=42)
#model = svm.SVC(gamma='scale')
#model = svm.LinearSVC()
#folds = GroupKFold(n_splits=10)
#scores = cross_val_score(model, X_new, y, cv=folds, groups=urls, scoring='f1_macro')
#print(scores)
#print("Our approach Feature Selection SVM F-measure: %s on average and %s SD" % (scores.mean(), scores.std()))
#features.append('diff_out_viewport_left')
#features.append('diff_out_viewport_right')
#selected_index = [ features[i] for i in range(len(features)) if rfecv.ranking_[i] == 1]
#print(selected_index)
