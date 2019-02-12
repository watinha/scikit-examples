import sys
from pipeline import BibParser, GenerateDataset

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
    BibParser(),
    GenerateDataset()
]

for action in actions:
    argument = action.execute(argument)

print(argument)

sys.exit(0)
