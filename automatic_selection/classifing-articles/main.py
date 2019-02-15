import sys

from sklearn.svm import LinearSVC

from pipeline import BibParser, GenerateDataset
from pipeline.classifier import DecisionTreeClassifier, LinearSVMClassifier, SVMClassifier, NaiveBayesClassifier
from pipeline.preprocessing import StopWordsFilter, PorterStemmerFilter, TextFilterComposite
from pipeline.transformation import LSATransformation
from pipeline.feature_selection import RFECVFeatureSelection, VarianceThresholdFeatureSelection

reload(sys)
sys.setdefaultencoding('utf-8')

argument = [
    'bibs/testing/round1-google.bib',
    'bibs/testing/round1-ieee.bib',
    'bibs/testing/round1-outros.bib',
    'bibs/testing/round2-google.bib',
    'bibs/testing/round2-ieee.bib',
    'bibs/testing/round2-outros.bib',
    'bibs/testing/round3-google.bib'
]
project_folder = 'testing'
#argument = [
#    'bibs/ontologies/round1-google.bib',
#    'bibs/ontologies/round1-ieee.bib',
#    'bibs/ontologies/round1-outros.bib',
#    'bibs/ontologies/round2-google.bib',
#    'bibs/ontologies/round2-ieee.bib',
#    'bibs/ontologies/round3-google.bib'
#]
#project_folder = 'ontologies'
#argument = [
#    'bibs/xbi/round1-google.bib',
#    'bibs/xbi/round1-ieee.bib',
#    'bibs/xbi/round1-outros.bib',
#    'bibs/xbi/round2-google.bib',
#    'bibs/xbi/round2-ieee.bib',
#    'bibs/xbi/round3-google.bib'
#]
#project_folder = 'xbis'

actions = [
    BibParser(write_files=False, project_folder=project_folder),
    TextFilterComposite([ StopWordsFilter(), PorterStemmerFilter() ]),
    GenerateDataset(ngram_range=(1,3)),
    LSATransformation(n_components=100, random_state=42),
#    VarianceThresholdFeatureSelection(threshold=0.0001),
#    RFECVFeatureSelection(LinearSVC()),
    DecisionTreeClassifier(42),
    SVMClassifier(42),
    LinearSVMClassifier(42),
    NaiveBayesClassifier(42)
]

for action in actions:
    argument = action.execute(argument)

sys.exit(0)
