import nltk
nltk.data.path.append('/Users/kohgiheon/share/nltk_data')
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np
import string

ps = PorterStemmer() #Normalization
stop_words = set(stopwords.words("english"))

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

age_indicator = ['year', 'years', 'old', 'year-old', 'month', 'months'] #indicator setting to find age
gender_indicator = ['man', 'male', 'woman', 'female', 'mal', 'femal'] #indicator setting to find gender
symptom_indicator = ['cm', 'present'] #indicator setting to find symptom
Disease_indicator = open('/Users/kohgiheon/Desktop/NLP_of_Clinical_data/practice/sample_diagnosis.txt').read() #indicator setting to find disease description
part_indicator = [' right', ' left',' transverse polyp'] #indicator setting to find body parts

ID_PATH = ['DE_ID_PAth_new_1.txt', 'De_ID_Path.txt', 'De_ID_Path_2.txt','De_ID_Path3.txt','De_ID_Path4.txt'] # test sample list ,'DE_ID_PAth_new_1.txt'
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

            elif catch in symptom_indicator or Disease_indicator :
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
    body = []
    proc = []
    Text_desc = []
    name = []
    MRN = [] #pre-setting for body, procedure, description, patient's name, and MRN

    freetext = open('/Users/kohgiheon/Desktop/NLP_of_Clinical_data/practice/pathology_medical_report_samples/' + pwd) #re-assign the file
    text = freetext.read()
    tokenized_text = word_tokenize(text) #tokenize the entire sample

    #Final Diagnosis
    Final_diagnosis = tokenized_text[tokenized_text.index('FINAL'):tokenized_text.index('Gross')] #extract sentence(s) for Final_diagnosis
    alphabet_index = list(string.ascii_uppercase) #pre-setting to find how many categories exist in Final_diagnosis
    for alphabet in alphabet_index : # find unexpected tokens whose value is already merged with any '.'
        if str(alphabet + '.') in str(Final_diagnosis) :
            Final_diagnosis[Final_diagnosis.index(str(alphabet + '.')) :Final_diagnosis.index(str(alphabet + '.'))+1] = alphabet, '.'

    alphabet = [] #find how many indice the Final_diagnosis has
    for word in Final_diagnosis :
        if word in alphabet_index :
            alphabet.append(word)

    for ind in alphabet : #process by index
        if ind in Final_diagnosis :
            if Final_diagnosis[Final_diagnosis.index(ind) + 1 ] in list(string.punctuation) : #if other Punctuation marks exist, unify them
                Final_diagnosis[Final_diagnosis.index(ind) + 1 ] = '.'
            if Final_diagnosis[Final_diagnosis.index(ind) +1] == '.' : #merge unnecessary information with '.' to blow out
                Final_diagnosis[Final_diagnosis.index(ind)] = ''.join(Final_diagnosis[Final_diagnosis.index(ind): Final_diagnosis.index(ind)+2])
                if Final_diagnosis[Final_diagnosis.index(ind + '.') +1] == '.' :
                    Final_diagnosis.pop(Final_diagnosis.index(ind + '.') +1)
            if ind != alphabet[-1] : #distinguish each index
                cut_by_index = Final_diagnosis[Final_diagnosis.index(ind +'.') +1 : Final_diagnosis.index(alphabet[alphabet.index(ind) +1])]
            elif ind == alphabet[-1] : #distinguish the last index
                cut_by_index = Final_diagnosis[Final_diagnosis.index(ind +'.') +1 : -1]

        Cword = tokenize_By_delimiter(cut_by_index) #tokenize the information which is given by index
        if Cword[1] in part_indicator : # if body part exists in sample, store information following such indice
            if join_with_space(Cword[0:2], ',') not in body :
                body.append(join_with_space(Cword[0:2], ','))
            if Cword[2] not in proc :
                proc.append(Cword[2])
            if Cword[3:] not in Text_desc :
                Text_desc.append(Cword[3:])
        elif Cword[1] not in part_indicator : # if body part does not exist in sample, store information following such indice
            if Cword[0] not in body :
                body.append(Cword[0])
            if Cword[1] not in proc :
                proc.append(Cword[1])
            if Cword[2:] not in Text_desc :
                Text_desc.append(Cword[2:])

    for i in range(int(len(Text_desc))) :
        keep_desc = []
        token_desc = word_tokenize(makeSentence(Text_desc[i])) #For the text desc list, make whole values as a string and tokenize again
        for word in token_desc : #take out any stop_words and Punctuation marks to simplify information
            if word in Disease_indicator and word not in list(stop_words) + list(string.punctuation) :
                keep_desc.append(word)
        Text_desc[i] = join_with_space(keep_desc, ' ')

    #Gross description
    Gross_dscp = tokenized_text[tokenized_text.index('Gross'):tokenized_text.index('Microscopic')] #Find sentence(s) for gross description

    name.append(join_with_space(Gross_dscp[Gross_dscp.index('name') + 2 : Gross_dscp.index('medical') - 1]," ")) #find name

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

    #merge
    if len(body) >= 2 :
        if len(proc) >=2 :
            for i in range(int(len(body))) :
                table= {"patient's name" : name, "Medical Record Number" : MRN, "gender" : gender,"body part" : body[i], "age" : age, "procedure" : proc[i], "Description" : Text_desc[i]}
                database_n = pd.DataFrame(table)
                database = database.append(database_n, ignore_index=True)
        else :
            for i in range(int(len(body))) :
                table= {"patient's name" : name, "Medical Record Number" : MRN, "gender" : gender,"body part" : body[i], "age" : age, "procedure" : proc, "Description" : Text_desc[i]}
                database_n = pd.DataFrame(table)
                database = database.append(database_n, ignore_index=True)
    else :
        if len(proc) >= 2 :
            for i in range(int(len(body))) :
                table= {"patient's name" : name, "Medical Record Number" : MRN, "gender" : gender,"body part" : body, "age" : age, "procedure" : proc[i], "Description" : join_with_space(Text_desc,'/')}
                database_n = pd.DataFrame(table)
                database = database.append(database_n, ignore_index=True)
        else :
            table= {"patient's name" : name, "Medical Record Number" : MRN, "gender" : gender,"body part" : body, "age" : age, "procedure" : proc, "Description" : join_with_space(Text_desc,'/')}
            database_n = pd.DataFrame(table)
            database = database.append(database_n, ignore_index=True)



print(database)
database.to_excel (r'/Users/kohgiheon/Desktop/patho_db.xlsx', index = None, header=True)

#
