import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

'''
By padding the inputs, we decide the maximum length of words in a sentence, then zero pads the rest, if the input length
is sharter than the designated length. 
'''
csv = 'data/clean_tweet.csv'
my_df = pd.read_csv(csv, index_col=0)
my_df.dropna(inplace=True)
my_df.reset_index(drop=True,inplace=True)
my_df = pd.concat([my_df[my_df.target == 0][:int(len(my_df)*0.05)],my_df[my_df.target == 4][:int(len(my_df)*0.05)]])
my_df.info()

x = my_df.text
y = my_df.target

from sklearn.cross_validation import train_test_split
SEED = 2000
x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)
x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)

print("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_train),
                                                                             (len(x_train[y_train == 0]) / (len(x_train)*1.))*100,
                                                                            (len(x_train[y_train == 1]) / (len(x_train)*1.))*100))
print("Validation set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_validation),
                                                                             (len(x_validation[y_validation == 0]) / (len(x_validation)*1.))*100,
                                                                            (len(x_validation[y_validation == 1]) / (len(x_validation)*1.))*100))
print("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_test),
                                                                             (len(x_test[y_test == 0]) / (len(x_test)*1.))*100,
                                                                            (len(x_test[y_test == 1]) / (len(x_test)*1.))*100))

from tqdm import tqdm
tqdm.pandas(desc = 'progress-bar')
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from sklearn import utils

def labelize_tweets_ug(tweets, label):
    result = []
    prefix = label
    for i,t in zip(tweets.index, tweets):
        result.append(TaggedDocument(t.split(),[prefix + '_%s' % i]))
    return result

all_x = pd.concat([x_train, x_validation,x_test])
all_x_w2v = labelize_tweets_ug(all_x,'all')

cores = multiprocessing.cpu_count()
model_ug_cbow = Word2Vec(sg = 0,size=100,negative=5,window=2,min_count=2,workers=cores, alpha=0.065,min_alpha=0.065)
model_ug_cbow.build_vocab([x.words for x in tqdm(all_x_w2v)])
for epoch in range(30):
    model_ug_cbow.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]),total_examples=len(all_x_w2v),epochs=1)
    model_ug_cbow.alpha -= 0.002
    model_ug_cbow.min_alpha = model_ug_cbow.alpha

model_ug_sg = Word2Vec(sg =1, size =100, negative=5,min_count=2,window=2,workers=cores,alpha=0.065,min_alpha=0.065)
model_ug_sg.build_vocab([x.words for x in tqdm(all_x_w2v)])
for epoch in range(30):
    model_ug_sg.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]),total_examples=len(all_x_w2v),epochs=1)
    model_ug_sg.alpha -= 0.002
    model_ug_sg.min_alpha = model_ug_sg.alpha

model_ug_cbow.save('word2vec_model/model_ug_cbow.word2vec')
model_ug_sg.save('word2vec_model/model_ug_sg.word2vec')


''' Preparation for CNN '''

from gensim.models import KeyedVectors
model_ug_cbow = KeyedVectors.load('word2vec_model/model_ug_cbow.word2vec')
model_ug_sg = KeyedVectors.load('word2vec_model/model_ug_sg.word2vec')

len(model_ug_cbow.wv.vocab.keys())

'''
By running below code block, I am constructing a sort of dictionary I can extract the word vectors from. 
Since I have two different Word2Vec models, below "embedding_index" will have concatenated vectors of the 
two models. For each model, I have 100 dimension vector representation of the words, and by concatenating 
each word will have 200 dimension vector representation.
'''
embeddings_index = {}
for w in model_ug_cbow.wv.vocab.keys():
    embeddings_index[w] = np.append(model_ug_cbow.wv[w],model_ug_sg.wv[w])
print('Found %s word vectors.' % len(embeddings_index))

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=100000)
tokenizer.fit_on_texts(x_train)
sequences = tokenizer.texts_to_sequences(x_train)

len(tokenizer.word_index)

## below are the first five entries of the original train data
for x in x_train[:5]:
    print(x)
sequences[:5]

## figure the max sequence length
length = []
for x in x_train:
    length.append(len(x.split()))

max(length) ## max length of sequence is 34, let's make the seq length is little longer 45
x_train_seq = pad_sequences(sequences,maxlen=45)
print('Shape of data tensor:', x_train_seq.shape)
x_train_seq[:5]

sequences_val = tokenizer.texts_to_sequences(x_validation)
x_val_seq = pad_sequences(sequences_val, maxlen=45)

num_words = 100000
embedding_matrix = np.zeros((num_words,200))
for word, i in tokenizer.word_index.items():
    if i >= num_words:
        continue
    embeddings_vector = embeddings_index.get(word)
    if embeddings_vector is not None:
        embedding_matrix[i] = embeddings_vector


''' Build Normal ANN Model '''
seed = 7
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten
from keras.layers.embeddings import Embedding

model_ptw2v = Sequential()
e = Embedding(100000,200,weights=[embedding_matrix],input_length=45,trainable=True)
model_ptw2v.add(e)
model_ptw2v.add(Flatten())
model_ptw2v.add(Dense(256,activation='relu'))
model_ptw2v.add(Dense(1,activation='sigmoid'))
model_ptw2v.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
model_ptw2v.fit(x_train_seq,y_train,validation_data=(x_val_seq,y_validation),epochs=5,batch_size=32,verbose=2)


''' CNN '''
from keras.layers import Conv1D,GlobalMaxPool1D
structure_test = Sequential()
e = Embedding(100000,200,input_length=45)
structure_test.add(e)
structure_test.add(Conv1D(filters=100,kernel_size=2,padding='valid',activation='relu',strides=1))
structure_test.summary()

model_cnn_01 = Sequential()
e = Embedding(100000, 200, weights=[embedding_matrix], input_length=45, trainable=True)
model_cnn_01.add(e)
model_cnn_01.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
model_cnn_01.add(GlobalMaxPool1D())
model_cnn_01.add(Dense(256, activation='relu'))
model_cnn_01.add(Dense(1, activation='sigmoid'))
model_cnn_01.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_cnn_01.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_validation), epochs=5, batch_size=32, verbose=2)

## combined bigram, trigram, fourgram
from keras.layers import Input,Dense, concatenate,Activation
from keras.models import Model

tweet_input = Input(shape=(45,),dtype='int32')

tweet_encoder = Embedding(100000,200,weights=[embedding_matrix],input_length=45,trainable=True)(tweet_input)
bigram_branch = Conv1D(filters=100,kernel_size=2,padding='valid',activation='relu',strides=1)(tweet_encoder)
bigram_branch = GlobalMaxPool1D()(bigram_branch)

trigram_branch = Conv1D(filters=100,kernel_size=3,padding='valid',activation='relu',strides=1)(tweet_encoder)
trigram_branch = GlobalMaxPool1D()(trigram_branch)

fourgram_branch = Conv1D(filters=100,kernel_size=4,padding='valid',activation='relu',strides=1)(tweet_encoder)
fourgram_branch = GlobalMaxPool1D()(fourgram_branch)

merged = concatenate(bigram_branch,trigram_branch,fourgram_branch)

merged = Dense(256,activation='relu')(merged)
merged = Dropout(0.2)(merged)
merged = Dense(1)(merged)
output = Activation('sigmoid')(merged)
model = Model(inputs=[tweet_input],output=[output])
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

from keras.callbacks import ModelCheckpoint
filepath = 'cnn_model/CNN_best_weights.{epoch:02d}-{val_acc:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath,monitor='val_acc',verbose=1,save_best_only=True, mode='max')
model.fit(x_train_seq,y_train, batch_size=32,epochs=5,validation_data=(x_val_seq,y_validation),callbacks=[checkpoint])


## load model
from keras.models import load_model
loaded_CNN_model = load_model('cnn_model/CNN_best_weights.{epoch:02d}-{val_acc:.4f}.hdf5')
loaded_CNN_model.evaluate(x=x_val_seq,y=y_validation)

## finally, model evluation with test set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
tvec = TfidfVectorizer(max_features=100000,ngram_range=(1,3))
tvec.fit(x_train)

x_train_tfidf = tvec.transform(x_train)
x_test_tfidf = tvec.transform(x_test)

lr_with_tfidf = LogisticRegression()
lr_with_tfidf.fit(x_train_tfidf,y_train)

lr_with_tfidf.score(x_test_tfidf,y_test)
yhat_lr = lr_with_tfidf.predict_proba(x_test_tfidf)



sequences_test = tokenizer.texts_to_sequences(x_test)
x_test_seq = pad_sequences(sequences_test,maxlen=45)
loaded_CNN_model.evaluate(x=x_test_seq, y=y_test)
yhat_cnn = loaded_CNN_model.predict(x_test_seq)


## plot the graph
fpr, tpr, threshold = roc_curve(y_test, yhat_lr[:,1])
roc_auc = auc(fpr, tpr)
fpr_cnn, tpr_cnn, threshold = roc_curve(y_test, yhat_cnn)
roc_auc_nn = auc(fpr_cnn, tpr_cnn)
plt.figure(figsize=(8,7))
plt.plot(fpr, tpr, label='tfidf-logit (area = %0.3f)' % roc_auc, linewidth=2)
plt.plot(fpr_cnn, tpr_cnn, label='w2v-CNN (area = %0.3f)' % roc_auc_nn, linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('Receiver operating characteristic: is positive', fontsize=18)
plt.legend(loc="lower right")
plt.show()



