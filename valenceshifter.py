
# coding: utf-8

# In[1]:


import pandas
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk import word_tokenize
from operator import itemgetter


# In[2]:


# Defining Contextual Valence Shifters
presuppositional_words = ['failure', 'barely', 'odd', 'stop','fix','last', 'hardly','recovering','recover']
connector_words = ['although', 'however', 'but', 'so', 'further', 'moreover', 'as well', 'nevertheless' ,'yet', 'instead']
negative_words = ['not', 'none', 'never', 'nobody', 'nowhere', 'neither', 'hit']
intensifier_words = ['very', 'lot', 'deep', 'amazingly', 'astoundingly','awful', 'bare',
                    'crazy', 'dreadfully', 'colossally', 'especially', 'exceptionally', 'excessively', 'extremely', 
                    'extraordinarily', 'fantastically', 'frightfully', 'fully', 'holy',
                    'incredibly', 'insanely', 'literally', 'mightily', 'moderately', 'outrageously',
                    'phenomenally', 'precious', 'quite', 'radically', 'rather', 'real', 'really', 
                    'remarkably', 'right', 'so', 'somewhat', 'super', 'supremely', 'surpassingly',
                    'terribly', 'terrifically', 'too', 'totally', 'uncommonly', 'unusually']
valence_shifters = presuppositional_words + connector_words + negative_words + intensifier_words


# In[3]:


# Creating dictionary
tag_dict  = {}

noun = ['NN','NNS','NNP','NNPS']
verb = ['VB','VBD','VBG','VBN','VBP','VBZ']
adj = ['JJ','JJR','JJS']
adverb = ['RB','RBR','RBS']
coordinating_conjunction = ['CC']
for word in noun :
    tag_dict[word] = 'n'
for word in verb :
    tag_dict[word] = 'v'
for word in adj :
    tag_dict[word] = 'a'
for word in adverb :
    tag_dict[word] = 'r'
for word in coordinating_conjunction :
    tag_dict[word] = 'c'


# In[4]:


# Flips the context of the next word with non-zero valence
def presuppositional (words, valence):
    for i in range (len(words)) :
        if (words[i] in presuppositional_words) :
            j = 0
            for j in range(i+1,len(words)):
                if (valence[j] != 0.0):
                    break
            if (j == len(words)):
                print('Incorrectly formed sentence')
                return 0
            valence[i] = -1 * valence[i]
            valence[j] = -1 * valence[j]
    return valence

# Words which make the previous part of the sentence have no effect
def connector (words, valence):
    for i in range (len(words)) :
        if (words [i] in connector_words):
            for j in range(i) :
                valence[j] = 0.0
    return valence

# Flips the context of the next word with non-zero valence
def negative (words, valence) :
    for i in range (len(words)) :
        if (words[i] in negative_words) :
            valence[i] = 0
            j = 0
            for j in range(i+1,len(words)):
                if (valence[j] != 0):
                    break
            if (j == len(words)):
                print('Incorrectly formed sentence')
                return 0
            valence[i] = -1 * valence[i]
            valence[j] = -1 * valence[j]
    return valence
            

# Enhances the valence of the next word with non-zero valence
def intensifier (words, valence):
    for i in range(len(words)):
        if (words[i] in intensifier_words) :
            j = 0
            for j in range(i+1,len(words)):
                if (valence[j] != 0):
                    break
            if (j == len(words)):
                print('Incorrectly formed sentence')
                return 0
            valence[j] = 1.5 * valence[j]
    return valence


# In[5]:


# Shifting the score of the sentence based on the corresponding valence shifter
def shift_score(words, valence):
    for word in words :
        if (word in presuppositional_words) :
            valence = presuppositional (words, valence)
        if (word in connector_words) :
            valence = connector (words, valence)
        if (word in negative_words) :
            valence = negative (words, valence)
        if (word in intensifier_words) :
            valence = intensifier(words, valence)   
    return valence


# In[6]:


# Filtering the tags that will be useful in synwordnet
def find_imp_words(tokens) :
    tags = nltk.pos_tag(tokens)
    imp_words = []
    for tag in tags:
        if (tag[1] in (tag_dict) or (tag[0] in valence_shifters)):
            imp_words.append(tag[0])
    return imp_words


# In[7]:


# Finding the net score of a sentence
def findscore(unimportant_tokens) :
    tokens = find_imp_words(unimportant_tokens)
    positive_score = 0.0
    negative_score = 0.0
    objective_score = 0.0
    valence = [0.0 for i in range(len(tokens))]
    i = 0
    for word in tokens:
        positive_word_score = 0
        negative_word_score = 0
        for item in list(swn.senti_synsets(word)):
            positive_word_score = positive_word_score + (item.pos_score())
            negative_word_score = negative_word_score + (item.neg_score())

        if (len(list(swn.senti_synsets(word))) != 0):
            positive_word_score = (positive_word_score/len(list(swn.senti_synsets(word))))
            negative_word_score = (negative_word_score/len(list(swn.senti_synsets(word))))

        valence[i] = (positive_word_score - negative_word_score)
        i = i + 1
    
    return tokens, valence


# In[8]:


inputdir = '/home/akshat/happy_nlp/happydb/'
data = pandas.DataFrame.from_csv(inputdir +'cleaned_hm.csv', index_col=None)
score = pandas.DataFrame([[data.iloc[0][0],data.iloc[0][3],0]], columns = ['wid','cleaned_hm','senti_score'])


# In[9]:


# Implementing the Algorithm: 1. Get the score 2. Shift context
result = []
for i in range (1000) :   #doing it for first 1000 happy moments
    valence = []
    data_object = data.iloc[i]
    sentence = data_object[3]
    tokens = nltk.word_tokenize(sentence)
    imp_tokens, valence = findscore(tokens)
    valence = shift_score(imp_tokens, valence)
    result.append((data_object[0], data_object[3], sum(valence) / len(valence)))


# In[10]:


sorted(result,key=itemgetter(2), reverse = True)


# In[11]:


def score_sentence (sentence) :
    valence = []
    tokens = nltk.word_tokenize(sentence)
    imp_tokens, valence = findscore(tokens)
    valence = shift_score(imp_tokens, valence)
    score = sum(valence) / len(valence)
    print("The score for the sentence is : ", score)


# In[12]:


score_sentence ('Scored two marvellous goals while playing football')

