import arff, np, random
from sklearn import tree, metrics
from sklearn.model_selection import cross_val_score, GroupKFold

features = [
    'size',
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

arff_file = arff.load(open('xbis.arff', 'rb'))
headers = [header[0] for header in arff_file['attributes']]
dataset = np.array(arff_file['data'])
features_index = map(lambda feature: headers.index(feature), features)
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

X = np.array(dataset[:, features_index], dtype=float)
X = np.concatenate((X, diff_out_viewport_left.T, diff_out_viewport_right.T), axis=1)
X_cross = np.array(dataset[:, crosscheck_index], dtype=float)
y = np.array(dataset[:, class_index], dtype=int)
urls = dataset[:, url_index]

random.seed(42)
model = tree.DecisionTreeClassifier(criterion='entropy', random_state=42)
folds = GroupKFold(n_splits=10)
scores = cross_val_score(model, X, y, cv=folds, groups=urls, scoring='f1_macro')
print(scores)
print("OUR APPROACH F-measure: %s on average and %s SD" % (scores.mean(), scores.std()))
random.seed(42)
model = tree.DecisionTreeClassifier(criterion='entropy', random_state=42)
folds = GroupKFold(n_splits=10)
scores = cross_val_score(model, X_cross, y, cv=folds, groups=urls, scoring='f1_macro')
print(scores)
print("CROSSCHECK   F-measure: %s on average and %s SD" % (scores.mean(), scores.std()))
