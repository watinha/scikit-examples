import re, sys, codecs

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

for file_index in range(6):
    filename = bib_filenames[file_index];
    with codecs.open(filename, 'r', encoding='utf-8') as bib_file:
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
            newfile = codecs.open('corpus/%s/%s-%d.txt' % (folder, insert, (bib_index + (file_index * 1000))), 'w', encoding='utf-8')
            print('from %s writing file %s/%s-%d.txt' % (filename, folder, insert, (bib_index + (file_index * 1000))))
            abstract = re.sub('[\n\r]', ' ', abstracts[bib_index][1])
            content = u'%s\n%s' % (titles[bib_index][1], abstract)
            newfile.write(content.decode())
            newfile.close()
        bib_file.close()

sys.exit(0)
