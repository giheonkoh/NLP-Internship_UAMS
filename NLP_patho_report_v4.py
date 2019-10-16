import nltk
nltk.data.path.append('/Users/kohgiheon/share/nltk_data')
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np
import string
import random

ps = PorterStemmer() #Normalization
stop_words = list(stopwords.words("english"))
# stop_words.pop(stop_words.index('of'))

def alphaPOS(list):
    keep = []
    for POS in list :
        if POS.isalpha() == True :
            keep.append(POS)
    return keep #identify letter in ASCII description

def makeSentence(list) :
    text = ''
    for element in list :
        text = str(text) + " " +  str(element)
    return text #make sentence with space

def Find_unique(list) :
    unique_list = []
    for x in list:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list #find unique values in a list

def join_with_space(list, delimiter) :
    lt = iter(list)
    string = str(next(lt))
    for i in lt :
        string += str(delimiter) + str(i)
    return string #merge elements in a list with a delimiter between the elements to make a string

def tokenize_By_delimiter(wlist) :
    str = ''
    words = []
    for word in wlist :
        if word not in list(string.punctuation) :
            str = str + ' ' + word
            if word == wlist[-1] :
                words.append(str)
        elif word in list(string.punctuation) :
            words.append(str)
            str = ''
    return words #merge values in a list except some delimiters presenting in the list

def Find_indice(list, obj) :
    ind = []
    ct = 0
    for word in list :
        if word == obj :
            ind.append(ct)
        ct +=1
    return ind

age_indicator = ['year', 'years', 'old', 'year-old', 'month', 'months'] #indicator setting to find age
gender_indicator = ['man', 'male', 'woman', 'female', 'mal', 'femal'] #indicator setting to find gender
symptom_indicator = ['cm', 'present', 'mm'] #indicator setting to find symptom
part_indicator = ['right', 'left','transverse polyp', 'sigmoid polyp', 'cecal polyp', 'right medial thigh', 'left axilla'] #indicator setting to find body parts
ID_PATH = ['De_ID_Path_2.txt'] #,'De_ID_Path_2.txt', 'De_ID_Path3.txt', 'De_ID_Path4.txt', 'DE_ID_PAth_new_1.txt','Test1.txt', 'Test2.txt', 'Test3.txt','Test4.txt','Test5.txt','Test6.txt','Test7.txt','Test8.txt','Test9.txt', 'Test10.txt','Test11.txt','Test12.txt','Test13.txt','Test14.txt','Test15.txt']
UMLS = open('/Users/kohgiheon/Desktop/NLP_of_Clinical_data/sample_diagnosis_UMLS.txt').read().lower() #UMLS LEXICON
DI = UMLS.split('\n')
print(DI[-2])
bool(any("vtti" in s for s in DI))

database = pd.DataFrame() #final database pre-setting
ID_PATH_count = 0
for pwd in ID_PATH : # for loop for processing all test sample at one click
    ID_PATH_count +=1
    print(ID_PATH_count)
    summary_all = pd.DataFrame() #temporary summary storage for detected words

    freetext = open('/Users/kohgiheon/Desktop/NLP_of_Clinical_data/practice/pathology_medical_report_samples/' + pwd) #open sample
    text = freetext.read() #read file
    sentences = sent_tokenize(text) #tokenize by senteces

    for sent in sentences : #process by sentence
        words = word_tokenize(sent) #tokenize by words while processing sentence by sentence
        filtered_sent = []
        for w in words: #stopwords
            if w not in stop_words :
                filtered_sent.append(w) #filtering out the stopwords

        Stemmed_words = []
        for w in filtered_sent: #Normalization
            Stemmed_words.append(ps.stem(w)) # Find and assign stemmed words

        tagged = nltk.pos_tag(Stemmed_words) # tagging part of speech to each word

        df = pd.DataFrame(tagged, columns = ["word", "POS"]) #make the tagged to data frame
        POS = alphaPOS(df['POS'].unique().tolist()) # exclude Punctuation Marks
        df['POS'] = df['POS'].astype('category') # make parts of speech to categorical variables

        CD = df[df['POS'] == 'CD'] #Find caridinal quantity(CD) which may be assumed as the meaningful
        CD_unit = df.iloc[CD.index +1]['word'] # Find words which is allocated to CD
        df_word= df['word'].tolist() # Make a list of words which is presented in a sentence

        unit_description = [] #pre-setting to store unit_description data
        for catch in CD_unit: #Extract caridinal quantity information with its unit and description
            if catch in age_indicator :
                if catch == 'year-old' :
                    cat = catch.split("-")
                    df_word[df_word.index(catch) : df_word.index(catch) +1] = cat
                    w = cat[0]
                    unit_description.append(df_word[df_word.index(w) +1])

                else : unit_description.append(df_word[df_word.index(catch) +1])

            elif catch in symptom_indicator or DI :
                text = makeSentence(df_word[df_word.index(catch)+1 : df_word.index('.')])
                unit_description.append(text)

        CD.loc[:,'unit'] = CD_unit.tolist() # modify information for unit
        CD.loc[:,'unit_description'] = unit_description #modify information for unit description
        CD = CD[['POS', 'word', 'unit', 'unit_description']] # rearrange the order of columns to merge with other information at a future step

        summary = CD.copy() # copy of CD to prevent unexpected change of information

        gender_presence = False
        for w in df.word : #for the words in a sentence, find the gender with using gender_indicator
            if w in gender_indicator :
                gender = df[df['word'] == w]
                if w in ['female', 'femal', 'lady', 'women', 'woman'] :
                    gender.loc[:,'word'] = 'female'
                else :
                    gender.loc[:,'word'] = 'male'
                gender.loc[:,'unit'] = str('gender')
                gender.loc[:,'unit_description'] = str('NA')
                gender_presence = True

        if gender_presence == True :
            gender = gender[['POS', 'word', 'unit', 'unit_description']] # rearrange the order of columns to merge with other information at a future step
            summary_all = pd.concat([summary_all, summary, gender]) # combine and store all information into temporary database
        else :
            summary_all = pd.concat([summary_all, summary]) # combine and store all information into temporary database

        if 'final' in words : #stop processing to start working on final diagnosis and gross description
            break

    # FINAL DIAGNOSIS & Gross Description
    freetext = open('/Users/kohgiheon/Desktop/NLP_of_Clinical_data/practice/pathology_medical_report_samples/' +pwd) #re-assign the file
    text = freetext.read()
    tokenized_text = word_tokenize(text) #tokenize the entire sample

    name = []

    #Final Diagnosis
    if tokenized_text[tokenized_text.index('Gross') -1] == '-' :
        Final_diagnosis = tokenized_text[tokenized_text.index('FINAL'):tokenized_text.index('.')]
    else :
        Final_diagnosis = tokenized_text[tokenized_text.index('FINAL'):tokenized_text.index('Gross')] #extract sentence(s) for Final_diagnosis
    alphabet_index = list(string.ascii_uppercase) #pre-setting to find how many categories exist in Final_diagnosis
    for alphabet in alphabet_index : # find unexpected tokens whose value is already merged with any '.'
        if str(alphabet + '.') in str(Final_diagnosis) :
            Final_diagnosis[Final_diagnosis.index(str(alphabet + '.')) :Final_diagnosis.index(str(alphabet + '.'))+1] = alphabet, '.'

    alphabet = [] #find how many indice the Final_diagnosis has
    for word in Final_diagnosis :
        if word in alphabet_index :
            alphabet.append(word)
    alphabet = Find_unique(alphabet)
    alphabet.sort()

    for i in alphabet :
        if i != 'A' :
            if abs(alphabet_index.index(i) - alphabet_index.index(alphabet[alphabet.index(i) -1])) >=2 :
                alphabet.pop(alphabet.index(i))

    for ind in alphabet : #process by index
        if ind in Final_diagnosis :
            if Final_diagnosis[Final_diagnosis.index(ind) + 1 ] in list(string.punctuation) : #if other Punctuation marks exist, unify them
                Final_diagnosis[Final_diagnosis.index(ind) + 1 ] = '.'
            if Final_diagnosis[Final_diagnosis.index(ind) +1] == '.' : #merge unnecessary information with '.' to blow out
                Final_diagnosis[Final_diagnosis.index(ind)] = ''.join(Final_diagnosis[Final_diagnosis.index(ind): Final_diagnosis.index(ind)+2])
                if Final_diagnosis[Final_diagnosis.index(ind + '.') +1] == '.' :
                    Final_diagnosis.pop(Final_diagnosis.index(ind + '.') +1)

    for word in Final_diagnosis :
        if word in alphabet_index :
            if Final_diagnosis[Final_diagnosis.index(word) +1] in list(string.punctuation) :
                Final_diagnosis[Final_diagnosis.index(word)] = ''.join(Final_diagnosis[Final_diagnosis.index(word): Final_diagnosis.index(word)+2])
                Final_diagnosis.pop(Final_diagnosis.index(str(word) + '.') +1)
    ind = Find_indice(Final_diagnosis, '-')
    Text_desc = []
    for i in ind :
        if i == ind[-1] :
            Text_desc.append(makeSentence(Final_diagnosis[i +1:]))
        else :
            Text_desc.append(makeSentence(Final_diagnosis[i+1:ind[ind.index(i) +1]]))

    ct = 0
    for desc in Text_desc :
        for a in alphabet :
            if str(a) + '.' in desc :
                ind = desc.index(str(a) + '.')
                Text_desc[ct] = desc[:ind]
            elif ',' in desc :
                ind = desc.index(',')
                Text_desc[ct] = desc[:ind]
        ct +=1

    grade = []
    i= 0
    while i < len(Text_desc) :
        if 'grade' in Final_diagnosis :
            ww = nltk.pos_tag(Final_diagnosis)
            if ww[Final_diagnosis.index('grade') + 1][1] == 'CD' :
                grade.append(Final_diagnosis[Final_diagnosis.index('grade') +1])
            else :
                grade.append(Final_diagnosis[Final_diagnosis.index('grade') -1])
        else :
            grade.append('NA')
        i +=1

    #body parts & proc
    cut_by_index = []
    for ind in alphabet :
        if str(ind) +'.' in Final_diagnosis :
            if ind != alphabet[-1] : #distinguish each index
                cut_by_index.append(makeSentence(Final_diagnosis[Final_diagnosis.index(str(ind) +'.') +1:
                Final_diagnosis.index(str(alphabet[alphabet.index(ind) +1]) +'.')]))
            elif ind == alphabet[-1] : #distinguish the last index
                cut_by_index.append(makeSentence(Final_diagnosis[Final_diagnosis.index(str(ind) +'.') +1: ]))

    body = []
    proc = []
    bp = ''
    for c in cut_by_index :
        No = False
        cc = word_tokenize(c)
        for i in cc :
            if join_with_space(cc[cc.index(i) : cc.index(i) +2], ' ') in part_indicator :
                ind = cc.index(i)
                cc[ind] = join_with_space(cc[ind : ind +2], ' ')
                bp = cc[ind]
                cc.pop(ind+1)
            elif join_with_space(cc[cc.index(i) : cc.index(i) +3], ' ') in part_indicator :
                ind = cc.index(i)
                cc[ind] = join_with_space(cc[ind : ind +3], ' ')
                bp = cc[ind]
                cc.pop(ind+1)
        for i in cc :
            if i in list(string.punctuation) :
                cc.pop(cc.index(i))
        for i in range(len(Text_desc)) :
            if Text_desc[i].strip() in c :
                if cc[1] in part_indicator or cc[2] in part_indicator:
                    if bp != '' :
                        body.append(join_with_space(cc[0:cc.index(bp) +1], ' '))
                        proc.append(cc[cc.index(bp) + int(len(bp.split())) -1])
                    else :
                        body.append(join_with_space(cc[0:2], ','))
                        proc.append(cc[2])
                elif cc[1] not in part_indicator :
                    body.append(cc[0])
                    for i in list(range(5)) :
                        if cc[1 +i] == '-' :
                            proc.append(cc[1])
                            No = True
                    if No == False:
                        proc.append(cc[1])

    coredesc = []
    for i in Text_desc :
        s = []
        a = word_tokenize(i)
        for wd in a:
            if wd.lower() in DI and str(wd) not in [s + '.' for s in alphabet ] :
                s.append(wd)
        coredesc.append(join_with_space(s, ' '))

    for i in coredesc :
        w = word_tokenize(i)
        for ww in w :
            if ww in string.punctuation or ww.lower() in list(stop_words) :
                w.pop(w.index(ww))
        if len(w) > 0 :
            coredesc[coredesc.index(i)] = join_with_space(w, ' ')
        else :
            coredesc[coredesc.index(i)] = 'NA'
    break

    desc = []
    errorrate = []
    for cd in coredesc :
        tokens = word_tokenize(cd)
        for w in tokens :
            for k in DI :
                if list(w)[0:len(w) -1] == list(k)[0:len(w) -1] :
                    if len(w) <= len(k) :
                        i = len(w)
                        for i in range(len(k)) :
                            add = []
                            for o in range(len(k) - len(w)) :
                                add.append(random.choice(string.ascii_lowercase))
                            w = makeSentence(list(w) + add)

                    if len(w) == len(k) :
                        pt=0
                        lw= list(w)
                        lk = list(k)
                        for i in range(len(w)) :
                            if lw[i] == lk[i] :
                                pt +=1
                        errorrate.append((w,k,float(pt/len(w))))

    #NOTE ID
    note_id  = tokenized_text[tokenized_text.index('Note_id') +2 : -1]

    #Gross description
    Gross_dscp = tokenized_text[tokenized_text.index('Gross'):tokenized_text.index('Microscopic')] #Find sentence(s) for gross description

    name.append(join_with_space(Gross_dscp[Gross_dscp.index('name') + 2 : Gross_dscp.index('medical') - 1]," ")) #find name

    MRN = []
    if "MRN" not in text : #find Medical Record Number
        Gross_dscp[Gross_dscp.index('medical'): Gross_dscp.index('number') +1] = [join_with_space(Gross_dscp[Gross_dscp.index('medical'): Gross_dscp.index('number') +1], " ")]
        if Gross_dscp[Gross_dscp.index('medical record number')+1] != ',' :
            MRN.append(Gross_dscp[Gross_dscp.index('medical record number')+1])
        else :
            MRN.append(Gross_dscp[Gross_dscp.index('medical record number')+2])

    if 'gender' in list(summary_all['unit']) :
        gender = summary_all[summary_all.unit == 'gender'].word.tolist() #find gender from summary_all data frame by finding the unit = gender
        gender = Find_unique(gender) #make unique
    else :
        gender = 'NA'

    if 'old' in list(summary_all.unit_description) :
        age = summary_all[summary_all.unit_description == 'old'].word.tolist() #find age from summary_all data frame by finding the unit_description = old
        age = Find_unique(age)
    else :
        age = 'NA'

    date = []
    for w in summary_all['word'] :
        if '/' in w :
            if w in Gross_dscp :
                date.append(w)
                break
            else :
                date.append(w)
                break
    #merge
    for i in range(int(len(body))) :
        table= {"patient's name" : name, "Medical Record Number" : MRN,
        "gender" : gender,"age" : age, "body part" : body[i],  "procedure" : proc[i],
        "Description" : coredesc[i],'Grade' : grade[i],'Date' : date, 'Reference' : note_id }
        database_n = pd.DataFrame(table)
        database = database.append(database_n, ignore_index=True)

    print('Done')

    # print(len(name), len(MRN), len(gender), len(age), len(body), len(proc), len(coredesc), len(grade), len(date), len(note_id))
    # print(name, MRN, gender, age, body, proc, coredesc, grade, date, note_id)
    # print(grade)

print(database)
database.to_excel (r'/Users/kohgiheon/Desktop/a.xlsx', index = None, header=True)

#
