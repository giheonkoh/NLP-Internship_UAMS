lexicon =open('/Users/kohgiheon/Desktop/LEXICON.txt').read()

import nltk
nltk.data.path.append('/Users/kohgiheon/share/nltk_data')
from nltk.tokenize import word_tokenize

def makeSentence(list) :
    text = ''
    for element in list :
        text = str(text) + " " +  str(element)
    return text

lex = []
for w in lexicon.splitlines() :
    if'base=' in w :
        ww = word_tokenize(w)
        lex.append(makeSentence(ww[2:]).strip())
        # print(w)

type(lex)
len(lex)
bool('Mycobilimbia fissuriseda' in lex)
