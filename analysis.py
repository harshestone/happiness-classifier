
import preprocess
import pandas
import operator
import numpy as np
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
import os


# Generates Word Cloud.
def get_image(alist):
    print("building word cloud")
    mpl.rcParams['font.size']=12               
    mpl.rcParams['savefig.dpi']=100          
    mpl.rcParams['figure.subplot.bottom']=.1 
    string=''
    for (a,b) in alist:
        for _ in range(b):
            string = string + str(a) + " "
    wordcloud = WordCloud(
                              background_color='white', collocations = False,
                              max_words=200,
                              max_font_size=40, 
                              random_state=42
                             ).generate(str(string))

  
    fig = plt.figure(1)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    fig.savefig("word1.png", dpi=1500)


# filters dataframe on basis of demographic details


def filter_demographic(attribute,val1,df,dataframe,val2 = -1):
    if(val2 == -1):
        df_1 = df.loc[(df[attribute]==val1)]
        wid_list = df_1.iloc[:,0]
       
    else:
        b=[]
        a = list(range(int(val1),int(val2)))
        for i in a:
            b.append(str(float(i)))
            b.append(str(i))
        df_1 = df.loc[(df[attribute].isin(b))]
        wid_list = df_1.iloc[:,0]
    wid_list = list(wid_list)
    df_2 = dataframe.loc[(dataframe['wid'].isin(wid_list))]
    return df_2


# query adding system and generates word cloud


def happy_words(dataframe,df_demo):
    
    while(1):
        print("Want to filter current data frame using demographic information Y or N")
        choice = input().strip().lower()
        if(choice == 'y'):
            print("1. Comparision filter")
            print("2. Equality filter")
            type = input()
            if(type=='1'):
                print("enter Attribute and upper bound and lower bound both exclusive")
                attribute = input().strip()
                upperbound = input().strip()
                lowerbound = input().strip()
                dataframe = filter_demographic(attribute,lowerbound,df_demo, dataframe,upperbound)
            if(type=='2'):
                print("enter Attribute and value")
                attribute = input().strip()
                value = input().strip()
                dataframe = filter_demographic(attribute,value, df_demo,dataframe)
            else:
                print("wrong choice entered")
                return
        if(choice !='y' and choice != 'n'):
            print("wrong choice entered")
            return
        print("1.Add query")
        print("2.View data frame")
        print("3.Exit")
        choice = input().strip()
        if(choice=='1'):
            print("1. Comparision query")
            print("2. Equality query")
            type = input().strip()
            if(type=='1'):
                print("enter Attribute and upper bound and lower bound both exclusive")
                attribute = input().strip()
                upperbound = input().strip()
                lowerbound = input().strip()
                dataframe = dataframe.loc[(dataframe[attribute]>lowerbound) and (dataframe[attribute]< upperbound)]
            if(type=='2'):
                print("enter Attribute and value")
                attribute = input().strip()
                value = input().strip()
                dataframe = dataframe.loc[(dataframe[attribute] == value)]
        if(choice == '2'):
            print("Wait for few seconds please")
            tokenized_moments,y_train,tokens = preprocess.get_tokenized_moments(dataframe,0,1)
            
            print("Building vocabulary")
            vocab = preprocess.make_dictionary(tokens,1)
            print("Almost done")
            sorted_vocab  =sorted(vocab.items(), key=operator.itemgetter(1), reverse = True)
            get_image(sorted_vocab[0:50])
            return
            
        if(choice == '3'):
            return 


inputdir = os.path.join(os.getcwd(),"happydb/")
# if tokens dictionary does not exists creat it.
if os.path.isfile(inputdir + '/tokens.txt'):
    print("building Tokens dictionary. This may take some time")
    preprocess.generate_tokens()

df_data = pandas.DataFrame.from_csv(inputdir +'cleaned_hm.csv', index_col=None)
df_demo = pandas.DataFrame.from_csv(inputdir +'demographic.csv', index_col=None)
happy_words(df_data,df_demo)





