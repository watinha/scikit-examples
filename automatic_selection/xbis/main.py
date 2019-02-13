import sys
from pipeline import BibParser, GenerateDataset
from pipeline.classifier import DecisionTreeClassifier, SVMClassifier, NaiveBayesClassifier
from pipeline.preprocessing import StopWordsFilter, PorterStemmerFilter, TextFilterComposite
from pipeline.transformation import LSATransformation

reload(sys)
sys.setdefaultencoding('utf-8')

argument = [
    'bibs/round1-google.bib',
    'bibs/round1-ieee.bib',
    'bibs/round1-outros.bib',
    'bibs/round2-google.bib',
    'bibs/round2-ieee.bib',
    'bibs/round3-google.bib'
]

actions = [
    BibParser(write_files=False),
    TextFilterComposite([ StopWordsFilter(), PorterStemmerFilter() ]),
    GenerateDataset(ngram_range=(1,3)),
    LSATransformation(n_components=100, random_state=42),
    DecisionTreeClassifier(42),
    SVMClassifier(42),
    NaiveBayesClassifier(42)
]

for action in actions:
    argument = action.execute(argument)

sys.exit(0)
