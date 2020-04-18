

import pandas
import json
import preprocess
import numpy as np
import keras
import os
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# runs LSTM
def lstm_run(X_train,y_train):             
    top_words = 20000
    # truncate and pad input sequences
    max_review_length = 100
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    y_train = np.array(y_train)
    #X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(7, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, nb_epoch=5, batch_size=64)
    return model

# convert data to be entered in lstm format
def lstm_convert_data(token_list,vocab):
    out_list=[]
    for item in token_list:
        int_list =[]
        for word in item:
            freq = vocab.get(word)
            if(freq == None):
                freq=0
            int_list.append(freq)
        out_list.append(int_list)
    return out_list
# categorize a particular happy moment
def categorize_this_happy_moment(happy_moment):
    input = preprocess.tokenize(happy_moment)
    input = lstm_convert_data([input],vocab)
    input = np.array(input)
    input = sequence.pad_sequences(input, maxlen=max_review_length)
    prediction = model.predict(input)
    max_index = np.argmax(np.array(prediction))
    print(predict_category.get(max_index))
    
inputdir = os.path.join(os.getcwd(),"happydb/")
if not os.path.isfile('tokens.txt'):
    print("building Tokens dictionary. This may take some time")
    preprocess.generate_tokens()

categories = {'achievement': [1,0,0,0,0,0,0], 'affection': [0,1,0,0,0,0,0], 'bonding': [0,0,1,0,0,0,0], 'enjoy_the_moment':[0,0,0,1,0,0,0], 'exercise':[0,0,0,0,1,0,0], 'leisure':[0,0,0,0,0,1,0],'nature':[0,0,0,0,0,0,1]}
df_data = pandas.DataFrame.from_csv(inputdir +'cleaned_hm.csv', index_col=None)
df_train = df_data.loc[(df_data['ground_truth_category'].notnull())]
print("Grabbing Happy Moments:")
tokenized_moments,y_train,train_tokens = preprocess.get_tokenized_moments(df_train)
vocab = preprocess.make_dictionary(train_tokens)
x_train = lstm_convert_data(tokenized_moments,vocab)
print("running LSTM")
model = lstm_run(x_train,np.array(y_train))
print("Predicting on new data")
pred_tokenized_moments,y,t = preprocess.get_tokenized_moments(df_data,1)
x_pred = lstm_convert_data(pred_tokenized_moments,vocab)
max_review_length = 100
x_pred = sequence.pad_sequences(x_pred, maxlen=max_review_length)
prediction = model.predict(x_pred)
predict_category = {0:'achievement', 1:'affection', 2:'bonding', 3:'enjoy_the_moment',4: 'exercise', 5:'leisure',6:'nature'}
cat_prediction=[]
for i in range(len(prediction)):
    max_index = np.argmax(np.array(prediction[i]))
    cat_prediction.append(predict_category.get(max_index))
df_data['our_prediction'] = pandas.Series(cat_prediction, index=df_data.index)
print("dataframe after prediction")
df_data
# write it in csv file
print("Writing dataframe to csv")
df_data.to_csv('after_prediction.csv')
print("done")









