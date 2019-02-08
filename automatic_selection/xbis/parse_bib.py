import re, sys

bib_filenames = [
    'round1-google.bib',
    'round1-ieee.bib',
    'round1-outros.bib',
    'round2-google.bib',
    'round2-ieee.bib',
    'round3-google.bib'
]

filename = bib_filenames[5];
with open(filename) as bib_file:
    bibfile = bib_file.read()
    titles = re.findall('(title)\s*=\s\{([^\}]*)\}', bibfile)
    abstracts = re.findall('(abstract)\s*=\s\{([^\}]*)\}', bibfile)
    inserir = re.findall('(inserir)\s*=\s\{([^\}]*)\}', bibfile)
    folder = filename.split('-')[0]

    if (len(titles) != len(abstracts) or len(titles) != len(inserir)):
        print 'Different number of titles, abstracts and inserir values...'
        sys.exit()

    for bib_index in range(len(titles)):
        insert = 'selecionado' if inserir[bib_index][1] == 'true' else 'removido'
        #newfile = open('%s/%s-%d.txt' % (folder, insert, bib_index))
        print('%s/%s-%d.txt' % (folder, insert, bib_index))
        content = '%s\n%s' % (titles[bib_index][1], abstracts[bib_index][1])
        print content
