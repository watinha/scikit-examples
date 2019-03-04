import sys

from sklearn import tree, naive_bayes
from sklearn.svm import LinearSVC, SVC

from pipeline import BibParser, GenerateDataset
from pipeline.classifier import DecisionTreeClassifier, LinearSVMClassifier, SVMClassifier, NaiveBayesClassifier, RandomForestClassifier
from pipeline.preprocessing import LemmatizerFilter, StopWordsFilter, PorterStemmerFilter, TextFilterComposite
from pipeline.transformation import LSATransformation
from pipeline.feature_selection import RFECVFeatureSelection, VarianceThresholdFeatureSelection
from pipeline.reporter import CSVReporter
from sklearn.feature_extraction.text import TfidfVectorizer

inputs = [
    {
        'argument': [ 'bibs/games/round1-todos.bib' ],
        'project_folder': 'games',
        'elimination_classifier': LinearSVC()
    },
    {
        'argument': [ 'bibs/slr/round1-todos.bib' ],
        'project_folder': 'slr',
        'elimination_classifier': tree.DecisionTreeClassifier()
    },
    {
        'argument': [ 'bibs/pair/round1-todos.bib' ],
        'project_folder': 'pair',
        'elimination_classifier': LinearSVC()
    },
   {
       'argument': [ 'bibs/illiterate/round1-others.bib' ],
       'project_folder': 'illiterate',
       'elimination_classifier': LinearSVC()
   },
   {
       'argument': [ 'bibs/mdwe/round1-acm.bib',
           'bibs/mdwe/round1-ieee.bib', 'bibs/mdwe/round1-sciencedirect.bib' ],
       'project_folder': 'mdwe',
       'elimination_classifier': LinearSVC()
   },
   {
       'argument': [ 'bibs/testing/round1-google.bib',
       'bibs/testing/round1-ieee.bib', 'bibs/testing/round1-outros.bib',
       'bibs/testing/round2-google.bib', 'bibs/testing/round2-ieee.bib',
       'bibs/testing/round2-outros.bib', 'bibs/testing/round3-google.bib'],
       'project_folder': 'testing',
       'elimination_classifier': tree.DecisionTreeClassifier()
   },
   {
       'argument': [ 'bibs/ontologies/round1-google.bib',
           'bibs/ontologies/round1-ieee.bib', 'bibs/ontologies/round1-outros.bib',
           'bibs/ontologies/round2-google.bib', 'bibs/ontologies/round2-ieee.bib',
           'bibs/ontologies/round3-google.bib' ],
       'project_folder': 'ontologies',
       'elimination_classifier': LinearSVC()
   },
   {
       'argument': [ 'bibs/xbi/round1-google.bib',
           'bibs/xbi/round1-ieee.bib', 'bibs/xbi/round1-outros.bib',
           'bibs/xbi/round2-google.bib', 'bibs/xbi/round2-ieee.bib',
           'bibs/xbi/round3-google.bib' ],
       'project_folder': 'xbis',
       'elimination_classifier': LinearSVC()
   }
]

reporter = CSVReporter('result/tf-idf-rfecv.csv')

for input in inputs:
    print(' ============================ ')
    print('   --- project %s ---' % (input['project_folder']))
    print(' ============================ ')
    project_folder = input['project_folder']
    argument = input['argument']
    elimination_classifier = input['elimination_classifier']
    actions = [
        BibParser(write_files=False, project_folder=project_folder),
        TextFilterComposite([ LemmatizerFilter(), StopWordsFilter(), PorterStemmerFilter() ]),
        GenerateDataset(TfidfVectorizer(ngram_range=(1,3), use_idf=True)),
        #LSATransformation(n_components=100, random_state=42),
        VarianceThresholdFeatureSelection(threshold=0.0001),
        RFECVFeatureSelection(elimination_classifier),
        DecisionTreeClassifier(seed=42, criterion='gini'),
        #RandomForestClassifier(seed=42, criterion='gini'),
        SVMClassifier(42),
        #LinearSVMClassifier(42),
        NaiveBayesClassifier(42),
        reporter
    ]

    for action in actions:
        argument = action.execute(argument)

reporter.report()
sys.exit(0)
