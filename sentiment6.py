import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas(desc = 'progress-bar')
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import multiprocessing
from sklearn import  utils
from sklearn.linear_model import LogisticRegression
''' Another Twitter sentiment analysis with Python — Part 6 (Doc2Vec) '''
'''
doc2vec: 
every paragraph is mapped to a unique vector, represented by a column in matrix D and every word is also mapped to 
a unique vector, represented by a column in matrix W. 
The paragraph vector and word vectors are averaged or concatenated to predict the next word in a context. 
The paragraph token can be thought of as another word. It acts as a memory that remembers what is missing from 
the current context — or the topic of the paragraph.
'''
csv = 'data/clean_tweet.csv'
my_df = pd.read_csv(csv, index_col = 0)
my_df.dropna(inplace=True)
my_df.reset_index(drop=True,inplace=True)
my_df.info()

x = my_df.text
y = my_df.target
from sklearn.cross_validation import train_test_split
SEED = 2000
x_train, x_v_t, y_train, y_v_t = train_test_split(x, y, test_size=.02, random_state=SEED)
x_validation, x_test, y_validation, y_test = train_test_split(x_v_t,y_v_t,test_size=.5, random_state=SEED)

'''
Below are the methods I used to get the vectors for each tweet.
1. DBOW 
2. DMC 
3. DMM 
4. DBOW + DMC
5. DBOW+DMM
'''
def labelize_tweets_ug(tweets,label):
    result = []
    prefix = label
    for i,t in zip(tweets.index, tweets):
        result.append(LabeledSentence(t.split(), [prefix + '_%s' %i]))
    return result

all_x = pd.concat([x_train,x_validation,x_test])
all_x_w2v = labelize_tweets_ug(all_x, 'all')

## DBOW
cores =  multiprocessing.cpu_count()
model_ug_dbow = Doc2Vec(dm= 0, vector_size = 100, negative=5, min_count = 2, workers=cores, alpha=0.065, min_alpha=0.065 )
model_ug_dbow.build_vocab([x for x in tqdm(all_x_w2v)])
for epoch in range(30):
    model_ug_dbow.train(utils.shuffle([x for x in tqdm(all_x_w2v)]),total_examples=len(all_x_w2v),epochs = 1)
    model_ug_dbow.alpha -= 0.002
    model_ug_dbow.min_alpha = model_ug_dbow.alpha

def get_vectors(model, corpus, size):
    vecs = np.zeros((len(corpus),size))
    n = 0
    for i in corpus.index:
        prefix = 'all_'+str(i)
        vecs[n] = model.docvecs[prefix]
        n+= 1
    return vecs

train_vecs_dbow = get_vectors(model_ug_dbow,x_train,100)
validation_vecs_dbow = get_vectors(model_ug_dbow,x_validation,100)

clf = LogisticRegression()
clf.fit(train_vecs_dbow,y_train)
clf.score(validation_vecs_dbow,y_train)

'''
Even though the DBOW model does not learn the meaning of the individual words, but as features to feed to a classifier,
it seems like it is doing its job. 
But the result does not seem to excel counter vectorizer or tfidf vectorizer. It might not be a direct comparison since 
either count vectorizer or tifidf vectorizer uses a large number of features to represent a tweet, but in this case, a 
vector for  each tweet has only 200 dimensions 
'''
model_ug_dbow.save('dov2vec_model/d2v_model_ug_dbow.doc2vec')
model_ug_dbow = Doc2Vec.load('dov2vec_model/d2v_model_ug_dbow.doc2vec')
model_ug_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)