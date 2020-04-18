
import pandas
import nltk
import operator
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import numpy as np
import os
from nltk.tokenize import sent_tokenize, word_tokenize
stemmer = SnowballStemmer("english")
import json
categories = {'achievement': [1,0,0,0,0,0,0], 'affection': [0,1,0,0,0,0,0], 
'bonding': [0,0,1,0,0,0,0], 'enjoy_the_moment':[0,0,0,1,0,0,0], 'exercise':[0,0,0,0,1,0,0], 
'leisure':[0,0,0,0,0,1,0],'nature':[0,0,0,0,0,0,1]}


# function to make dictionary out of tokens
def make_dictionary(tokens,option=0) :
    vocab={}
    stopWords=[]
    if(option==1):
        stopWords = list(stopwords.words('english')) + ["'nt"]
    for word in tokens:
        if word not in stopWords:
            if word in vocab.keys() :
                vocab[word] = vocab[word] + 1
            else :
                vocab[word] = 1
    return vocab


#  function to get tokenized moments of dataframe passed from token dictionary created.

def get_tokenized_moments(df,option=0,flag = 0):
    tokenized_moments=[]
    train_tokens=[]
    y_train=[]
    hmid_list = df.iloc[:,0]
    token_dictionary = json.load(open("tokens.txt"))
    output_dictionary = json.load(open("outputs.txt"))
    for hmid in hmid_list:
        try:
            token = token_dictionary[str(hmid)]
            if(flag==1):
                token = list(set(token))
            tokenized_moments.append(token)
            if(option ==0):
                y_train.append(categories.get((output_dictionary[str(hmid)])))
                train_tokens = train_tokens + token
        except:
            continue
    return tokenized_moments,y_train,train_tokens


# function to tokenize and stem . option is set 1 when we want to remove stop words.


def tokenize(paras,option=0):
    tokens=[]
    filtered_tokens=[]
    if(option==1):
        stopWords = set(stopwords.words('english'))
    else:
        stopWords=[]
    delimiters =[".", ",", ";", ":", "?", "/", "!", "'s", "'ll", "'d", ")", "("]
    tokens = word_tokenize(paras.lower())
    for word in tokens:
        if word not in ['\n','\t'] and word not in delimiters and word not in stopWords :
            filtered_tokens= filtered_tokens + [word]
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


# generates dictionary of tokens for future uses 


def generate_tokens():
    inputdir = os.path.join(os.getcwd(),"happydb/")
    df_data = pandas.DataFrame.from_csv(inputdir +'cleaned_hm.csv',index_col=None)
    token_dictionary={ }
    output_dictionary = {}
    for i in range (len(df_data)) :
        line = df_data.iloc[i][4]
       
        hmid = df_data.iloc[i][0]
        token = tokenize(line)
        token_dictionary[str(hmid)]=token
        category = df_data.iloc[i][7]
        output_dictionary[str(hmid)]=category
    json.dump(token_dictionary, open("tokens.txt",'w'))
    json.dump(output_dictionary, open("outputs.txt",'w'))
    
   

