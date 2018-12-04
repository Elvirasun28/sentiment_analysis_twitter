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
my_df = pd.concat([my_df[my_df.target == 0][:int(len(my_df)*0.1)],my_df[my_df.target == 4][:int(len(my_df)*0.1)]])
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
