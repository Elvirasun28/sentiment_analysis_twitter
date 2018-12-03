import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from tqdm import tqdm
tqdm.pandas(desc ='progress-bar')
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import multiprocessing
from sklearn import utils
from sklearn.linear_model import  LogisticRegression
from gensim.models.phrases import Phrases
from gensim.models.phrases import Phraser
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import  RidgeClassifier, PassiveAggressiveClassifier,Perceptron
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from datetime import time

'''Another Twitter sentiment analysis with Python — Part 7 (Phrase modeling + Doc2Vec) '''
plt.style.use('fivethirtyeight')
csv = 'data/clean_tweet.csv'
my_df = pd.read_csv(csv, index_col=0)

my_df.dropna(inplace=True)
my_df.reset_index(drop=True,inplace=True)
my_df = pd.concat([my_df[my_df.target == 0][:int(len(my_df)*0.05)],my_df[my_df.target == 4][:int(len(my_df)*0.05)]])
my_df.info()
SEED = 1
## train/dev/test split
x = my_df.text
y = my_df.target
x_train, x_validation_test,y_train, y_validation_test = train_test_split(x,y,test_size=.02, random_state=SEED)
x_validation,x_test, y_validation, y_test = train_test_split(x_validation_test,y_validation_test,test_size=.5, random_state=SEED)
print("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_train),
                                                                             (len(x_train[y_train == 0]) / (len(x_train)*1.))*100,
                                                                            (len(x_train[y_train == 4]) / (len(x_train)*1.))*100))
print("Validation set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_validation),
                                                                             (len(x_validation[y_validation == 0]) / (len(x_validation)*1.))*100,
                                                                            (len(x_validation[y_validation == 4]) / (len(x_validation)*1.))*100))
print("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_test),
                                                                             (len(x_test[y_test == 0]) / (len(x_test)*1.))*100,
                                                                            (len(x_test[y_test == 4]) / (len(x_test)*1.))*100))

def get_vectors(model,corpus,size):
    vecs = np.zeros((len(corpus),size))
    n = 0
    for i in corpus.index:
        prefix = 'all_'+str(i)
        vecs[n] = model.docvecs[prefix]
        n+= 1
    return vecs

def get_concat_vectors(model1,model2,corpus, size):
    vecs = np.zeros((len(corpus),size))
    n = 0
    for i in corpus.index:
        prefix = 'all_'+str(i)
        vecs[n] = np.append(model1.docvecs[prefix],model2.docvecs[prefix])
        n += 1
    return vecs


## phrasing model
tokenized_train = [t.split() for t in x_train]
phrases = Phrases(tokenized_train)
bigram = Phraser(phrases)


### ex1
sent = [u'the', u'mayor', u'of', u'new', u'york', u'was', u'there']
print(bigram[sent])
### ex2
x_train[2]
bigram[x_train[2].split()]

## transform our corpus with this bigram model
def labelize_tweets_bg(tweets, label):
    result = []
    prefix = label
    for i,t in zip(tweets.index, tweets):
        result.append(LabeledSentence(bigram[t.split()],[prefix+'_%s' %i]))
    return result

all_x = pd.concat([x_train,x_validation,x_test])
all_x_w2v_bg = labelize_tweets_bg(all_x, 'all')

## DBOW bigram
cores = multiprocessing.cpu_count()
model_bg_dbow = Doc2Vec(dm=0,vector_size=100,negative=5,min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_bg_dbow.build_vocab([x for x in tqdm(all_x_w2v_bg)])

for epoch in range(30):
    model_bg_dbow.train(utils.shuffle([x for x in tqdm(all_x_w2v_bg)]),total_examples=len(all_x_w2v_bg),epochs=1)
    model_bg_dbow.alpha -= 0.002
    model_bg_dbow.min_alpha = model_bg_dbow.alpha

train_vecs_dbow_bg =  get_vectors(model_bg_dbow,x_train, 100)
validation_vecs_dbow_bg = get_vectors(model_bg_dbow,x_validation,100)

clf = LogisticRegression()
clf.fit(train_vecs_dbow_bg,y_train)
clf.score(validation_vecs_dbow_bg,y_validation)
model_bg_dbow.save('doc2vec_model/d2v_model_bg_dbow.doc2vec')
model_bg_dbow = Doc2Vec.load('doc2vec_model/d2v_model_bg_dbow.doc2vec')
model_bg_dbow.delete_temporary_training_data(keep_doctags_vectors=True,keep_inference=True)

# DMC Bigram
cores = multiprocessing.cpu_count()
model_bg_dmc = Doc2Vec(dm = 1, dm_concat=1,vector_size=100,window=2,negative=5,min_count=2,workers=cores,alpha=0.065,min_alpha=0.065)
model_bg_dmc.build_vocab([x for x in tqdm(all_x_w2v_bg)])
for epoch in range(30):
    model_bg_dmc.train(utils.shuffle([x for x in tqdm(all_x_w2v_bg)]), total_examples=len(all_x_w2v_bg), epochs=1)
    model_bg_dmc.alpha -= 0.002
    model_bg_dmc.min_alpha = model_bg_dmc.alpha

model_bg_dmc.most_similar('new_york')
train_vecs_dmc_bg = get_vectors(model_bg_dmc,x_train,100)
validation_vecs_dmc_bg = get_vectors(model_bg_dmc,x_validation,100)
clf = LogisticRegression()
clf.fit(train_vecs_dmc_bg,y_train)
clf.score(validation_vecs_dmc_bg,y_validation)
model_bg_dmc.save('doc2vec_model/d2v_model_bg_dmc.doc2vec')
model_bg_dmc = Doc2Vec.load('doc2vec_model/d2v_model_bg_dmc.doc2vec')
model_bg_dmc.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

## DMM Bigram
cores = multiprocessing.cpu_count()
model_bg_dmm = Doc2Vec(dm=1,dm_mean=1,vector_size=100,window=4,negative=5,min_count=2,workers=cores,alpha=0.065,min_alpha=0.065)
model_bg_dmm.build_vocab([x for x in tqdm(all_x_w2v_bg)])
for epoch in range(30):
    model_bg_dmm.train(utils.shuffle([x for x in tqdm(all_x_w2v_bg)]), total_examples=len(all_x_w2v_bg), epochs=1)
    model_bg_dmm.alpha -= 0.002
    model_bg_dmm.min_alpha = model_bg_dmm.alpha

train_vecs_dmm_bg = get_vectors(model_bg_dmm,x_train,100)
validation_vecs_dmm_bg = get_vectors(model_bg_dmm,x_validation,100)
clf = LogisticRegression()
clf.fit(train_vecs_dmm_bg,y_train)
clf.score(validation_vecs_dmm_bg,y_validation)
model_bg_dmm.save('doc2vec_model/d2v_model_bg_dmm.doc2vec')
model_bg_dmm = Doc2Vec.load('doc2vec_model/d2v_model_bg_dmm.doc2vec')
model_bg_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)


## combined models
train_vecs_dbow_dmc_bg = get_concat_vectors(model_bg_dbow,model_bg_dmc,x_train,200)
validation_vecs_dbow_dmc_bg = get_concat_vectors(model_bg_dbow,model_bg_dmc,x_validation,200)
clf = LogisticRegression()
clf.fit(train_vecs_dbow_dmc_bg,y_train)
clf.score(validation_vecs_dbow_dmc_bg,y_validation)

train_vecs_dbow_dmm_bg = get_concat_vectors(model_bg_dbow,model_bg_dmm,x_train,200)
validation_vecs_dbow_dmm_bg = get_concat_vectors(model_bg_dbow,model_bg_dmm,x_validation,200)
clf = LogisticRegression()
clf.fit(train_vecs_dbow_dmm_bg,y_train)
clf.score(validation_vecs_dbow_dmm_bg,y_validation)



## Trigram
tg_phrases = Phrases(bigram[tokenized_train])
trigram = Phraser(tg_phrases)

trigram[bigram[x_train[3].split()]]

def labelize_tweets_tg(tweets,label):
    result = []
    prefix = label
    for i, t in zip(tweets.index, tweets):
        result.append(LabeledSentence(trigram[bigram[t.split()]], [prefix + '_%s' % i]))
    return result

all_x = pd.concat([x_train,x_validation,x_test])
all_x_w2v_tg = labelize_tweets_tg(all_x, 'all')

## DBOW bigram
cores = multiprocessing.cpu_count()
model_tg_dbow = Doc2Vec(dm=0,vector_size=100,negative=5,min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_tg_dbow.build_vocab([x for x in tqdm(all_x_w2v_tg)])

for epoch in range(30):
    model_tg_dbow.train(utils.shuffle([x for x in tqdm(all_x_w2v_tg)]),total_examples=len(all_x_w2v_tg),epochs=1)
    model_tg_dbow.alpha -= 0.002
    model_tg_dbow.min_alpha = model_tg_dbow.alpha

train_vecs_dbow_tg =  get_vectors(model_tg_dbow,x_train, 100)
validation_vecs_dbow_tg = get_vectors(model_tg_dbow,x_validation,100)

clf = LogisticRegression()
clf.fit(train_vecs_dbow_tg,y_train)
clf.score(validation_vecs_dbow_tg,y_validation)
model_tg_dbow.save('doc2vec_model/d2v_model_tg_dbow.doc2vec')
model_tg_dbow = Doc2Vec.load('doc2vec_model/d2v_model_tg_dbow.doc2vec')
model_tg_dbow.delete_temporary_training_data(keep_doctags_vectors=True,keep_inference=True)

# DMC Trigram
cores = multiprocessing.cpu_count()
model_tg_dmc = Doc2Vec(dm = 1, dm_concat=1,vector_size=100,window=2,negative=5,min_count=2,workers=cores,alpha=0.065,min_alpha=0.065)
model_tg_dmc.build_vocab([x for x in tqdm(all_x_w2v_tg)])
for epoch in range(30):
    model_tg_dmc.train(utils.shuffle([x for x in tqdm(all_x_w2v_tg)]), total_examples=len(all_x_w2v_tg), epochs=1)
    model_tg_dmc.alpha -= 0.002
    model_tg_dmc.min_alpha = model_tg_dmc.alpha

model_tg_dmc.most_similar('new_york')
train_vecs_dmc_tg = get_vectors(model_tg_dmc,x_train,100)
validation_vecs_dmc_tg = get_vectors(model_tg_dmc,x_validation,100)
clf = LogisticRegression()
clf.fit(train_vecs_dmc_tg,y_train)
clf.score(validation_vecs_dmc_tg,y_validation)
model_tg_dmc.save('doc2vec_model/d2v_model_tg_dmc.doc2vec')
model_tg_dmc = Doc2Vec.load('doc2vec_model/d2v_model_tg_dmc.doc2vec')
model_tg_dmc.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

## DMM Trigram
cores = multiprocessing.cpu_count()
model_tg_dmm = Doc2Vec(dm=1,dm_mean=1,vector_size=100,window=4,negative=5,min_count=2,workers=cores,alpha=0.065,min_alpha=0.065)
model_tg_dmm.build_vocab([x for x in tqdm(all_x_w2v_tg)])
for epoch in range(30):
    model_tg_dmm.train(utils.shuffle([x for x in tqdm(all_x_w2v_tg)]), total_examples=len(all_x_w2v_tg), epochs=1)
    model_tg_dmm.alpha -= 0.002
    model_tg_dmm.min_alpha = model_tg_dmm.alpha

train_vecs_dmm_tg = get_vectors(model_tg_dmm,x_train,100)
validation_vecs_dmm_tg = get_vectors(model_tg_dmm,x_validation,100)
clf = LogisticRegression()
clf.fit(train_vecs_dmm_tg,y_train)
clf.score(validation_vecs_dmm_tg,y_validation)
model_tg_dmm.save('doc2vec_model/d2v_model_tg_dmm.doc2vec')
model_tg_dmm = Doc2Vec.load('doc2vec_model/d2v_model_tg_dmm.doc2vec')
model_tg_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

## combined trigram

train_vecs_dbow_dmc_tg  = get_concat_vectors(model_tg_dbow,model_tg_dmc, x_train, 200)
validation_vecs_dbow_dmc_tg = get_concat_vectors(model_tg_dbow,model_tg_dmc, x_validation, 200)
clf = LogisticRegression()
clf.fit(train_vecs_dbow_dmc_tg, y_train)
clf.score(validation_vecs_dbow_dmc_tg, y_validation)

train_vecs_dbow_dmm_tg = get_concat_vectors(model_tg_dbow,model_tg_dmm, x_train, 200)
validation_vecs_dbow_dmm_tg = get_concat_vectors(model_tg_dbow,model_tg_dmm, x_validation, 200)
clf = LogisticRegression()
clf.fit(train_vecs_dbow_dmm_tg, y_train)
clf.score(validation_vecs_dbow_dmm_tg, y_validation)

## bigram DBOW + trigram DMM:
model_bg_dbow = Doc2Vec.load('doc2vec_model/d2v_model_bg_dbow.doc2vec')
model_tg_dmm = Doc2Vec.load('doc2vec_model/d2v_model_tg_dmm.doc2vec')
model_bg_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
model_tg_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
train_vecs_bgdbow_tgdmm  = get_concat_vectors(model_bg_dbow,model_tg_dmm, x_train, 200)
validation_vecs_bgdbow_tgdmm = get_concat_vectors(model_bg_dbow,model_tg_dmm, x_validation, 200)
clf = LogisticRegression()
clf.fit(train_vecs_bgdbow_tgdmm, y_train)
clf.score(validation_vecs_bgdbow_tgdmm, y_validation)




from sklearn.preprocessing import MinMaxScaler
mnscaler = MinMaxScaler()
d2v_bgdbow_tgdmm_mm = mnscaler.fit_transform(train_vecs_bgdbow_tgdmm)
d2v_bgdbow_tgdmm_mm_val = mnscaler.fit_transform(validation_vecs_bgdbow_tgdmm)

names1 = ['Logistic Regression','Multinomial NB','Bernoulli NB','Ridge Classifier','Perceptron','Passive-Aggressive','Nearest Centoid']
classifiers1 = [
    LogisticRegression(),
    MultinomialNB(),
    BernoulliNB(),
    RidgeClassifier(),
    Perceptron(),
    PassiveAggressiveClassifier(),
    NearestCentroid()
    ]
zipped_clf1 = zip(names1,classifiers1)
from time import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
def classifier_comparator_d2v(train_vectors,validation_vectors,classifier=zipped_clf1):
    result = []
    for n ,c in classifier:
        checker_pipeline = Pipeline([
            ('classifier',c)
        ])
        print('Validation result for {}'.format(n))
        print(c)
        clf_accuracy, tt_time = accuracy_summary(checker_pipeline,train_vectors, y_train,validation_vectors, y_validation)
        result.append((n,clf_accuracy,tt_time))
    return result

classifier_comparator_d2v(d2v_bgdbow_tgdmm_mm,d2v_bgdbow_tgdmm_mm_val)

