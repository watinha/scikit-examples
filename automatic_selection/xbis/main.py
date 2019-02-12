import sys
from pipeline import BibParser

reload(sys)
sys.setdefaultencoding('utf-8')

bib_filenames = [
    'bibs/round1-google.bib',
    'bibs/round1-ieee.bib',
    'bibs/round1-outros.bib',
    'bibs/round2-google.bib',
    'bibs/round2-ieee.bib',
    'bibs/round3-google.bib'
]

actions = [
    BibParser(bib_filenames)
]

for action in actions:
    action.execute()

sys.exit(0)
