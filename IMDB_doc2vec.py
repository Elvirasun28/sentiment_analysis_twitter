import locale
import glob
import os.path
import requests
import sys
import tarfile
import codecs
from smart_open import smart_open
import re

### load dataset
dirname = 'data/aclImdb'
filename = 'data/aclImdb/aclImdb_v1.tar.gz'
locale.setlocale(locale.LC_ALL,'C')
all_lines = []
if sys.version > '3':
    control_chars = [chr(0x85)]
else:
    control_chars = [unichr(0x85)]

# convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text):
    norm_text = text.lower()
    # replace breaks with spaces
    norm_text = norm_text.replace('<br />',' ')
    # pad punctuation with spaces on both sides
    norm_text = re.sub(r"([\.\",\(\)!\?;:])",'\\l',norm_text)
    return norm_text

if not os.path.isfile('data/aclImdb/alldata-id.txt'):
    if not os.path.isdir(dirname):
        if not os.path.isfile(filename):
            # download IMDB archive
            print('Downloading IMDB archive........')
            url = u'http://ai.stanford.edu/~amaas/data/sentiment/' + filename
            r = requests.get(url)
            with smart_open(filename,'wb') as f:
                f.write(r.content)
        tar = tarfile.open(filename, mode='r')
        tar.extractall()
        tar.close()
    else:
        print('IMDB archive directory already available without download.')


# collect & normalize test/train data
print('Cleaning up dataset ......')
folders = ['train/pos', 'train/neg', 'test/pos', 'test/neg', 'train/unsup']
for fol in folders:
    temp = u''
    newline = '\n'.encode('utf-8')
    output = fol.replace('/','-')+'.txt'
    # is there a better pattern to use?
    txt_files = glob.glob(os.path.join(dirname, fol, '*.txt'))
    print(" %s: %i files" % (fol, len(txt_files)))
    with smart_open(os.path.join(dirname,output),'wb') as n:
        for i,txt in enumerate(txt_files):
            with smart_open(txt,'rb') as t:
                one_text = t.read().decode('utf-8')
                for c in control_chars:
                    one_text = one_text.replace(c,' ')
                one_text = normalize_text(one_text)
                all_lines.append(one_text)
                n.write(one_text.encode('utf-8'))
                n.write(newline)

with smart_open(os.path.join(dirname, 'alldata-id.txt'),'wb') as f:
    for idx, line in enumerate(all_lines):
        num_line = u"_*{0} {1}\n".format(idx, line)
        f.write(num_line.encode('utf-8'))

assert os.path.isfile("data/aclImdb/alldata-id.txt"), "alldata-id.txt unavailable"
print("Success, alldata-id.txt is available for next steps.")


##
import gensim
from gensim.models.doc2vec import TaggedDocument
from collections import namedtuple
SentimentDocument = namedtuple('SentimentDocument','words tags split sentiment')

alldocs = []
with smart_open('data/aclImdb/alldata-id.txt','rb', encoding='utf-8') as alldata:
    for line_no,line in enumerate(alldata):
        tokens = gensim.utils.to_unicode(line).split()
        words = tokens[1:]
        tags = [line_no]
        split = ['train','test','extra','extra'][line_no // 25000] # 25k train, 25k test, 50k extra
        sentiment = [1.0,0.0,1.0,0.0,None,None,None,None][line_no// 12500]  # [12.5K pos, 12.5K neg]*2 then unknown
        alldocs.append(SentimentDocument(words,tags,split,sentiment))

train_docs = [doc for doc in alldocs if doc.split == 'train']
test_docs = [doc for doc in alldocs if doc.split == 'test']
print('%d docs: %d train-sentiment, %d test-sentiment' % (len(alldocs), len(train_docs), len(test_docs)))


from random import shuffle
doc_list = alldocs[:]
shuffle(doc_list)


## set-up doc2vec training & evaluation models
from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import multiprocessing

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, 'This will be painfully slow otherwise'

simple_models = [
    # PV-DBOW plain
    Doc2Vec(dm=0, vector_size=100,negative=5,hs=0,min_count = 2, sample = 0, epochs=20,workers=cores),
    # PV-DM w/ default averaging, a higher starting alpha may improve CBOW/PV-DMmodes
    Doc2Vec(dm=1, vector_size=100,window=2,negative=5,hs=0,min_count=2,sample=0,epochs=20,workers=cores,alpha=0.05,comment='alpha==0.05'),
    # PV-DM w/concatenation -big,slow,experimental model
    Doc2Vec(dm=1,dm_concat = 1,vector_size=100,window=2,negative=5,hs=0,min_count=2,sample=0,epochs=20,workers=cores)
]

for model in simple_models:
    model.build_vocab(alldocs)
    print("%s vocabulary scanned & state initialized" % model)

models_by_name = OrderedDict((str(model),model) for model in simple_models)

## combine a pararaph vector from distributed bag of word and distributed memory improves performance
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[0],simple_models[1]])
models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[0],simple_models[2]])


## predictive evaluation methods - LR
import numpy as np
import statsmodels.api as sm
from random import sample

def logistic_predictor_from_data(train_tgt,train_reg):
    logit = sm.Logit(train_tgt,train_reg)
    pred = logit.fit(disp=0)
    return pred

def error_rate_for_model(test_model, train_set, test_set, reinfer_train=False, reinfer_test = False,
                         infer_steps=None,infer_alpha=None,infer_subsample=0.2):
    ''' Report error rate on test_doc sentiments, using supplied model and train_docs '''
    train_tgt = [doc.sentiment for doc in train_set]
    if reinfer_train:
        train_reg = [test_model.infer_vector(doc.words, step = infer_steps, alpha=infer_alpha) for doc in train_set]
    else:
        train_reg = [test_model.docvecs[doc.tags[0]] for doc in train_set]
    train_reg = sm.add_constant(train_reg)
    pred = logistic_predictor_from_data(train_tgt,train_reg)

    test_data = test_set
    if reinfer_test:
        if infer_subsample < 1.0:
            test_data = sample(test_data,int(infer_subsample * len(test_data)))
        test_reg = [test_model.infer_vector(doc.words, steps = infer_steps,alpha=infer_alpha) for doc in test_set]
    else:
        test_reg = [test_model.docvecs[doc.tags[0]] for doc in test_docs]
    test_reg = sm.add_constant(test_reg)

    # predict & evaluate
    test_pred = pred.predict(test_reg)
    corrects = sum(np.rint(test_pred) == [doc.sentiment for doc in test_data])
    errors = len(test_pred) - corrects
    error_rate = float(errors) / len(test_pred)
    return (error_rate, errors, len(test_pred), pred)


from collections import defaultdict
error_rates = defaultdict(lambda: 1.0)

for model in simple_models:
    print('Training %s' % model)
    model.train(doc_list, total_examples=len(doc_list), epochs=model.epochs)
    print('\nEvaluating %s' % model)
    err_rate, err_count, test_count, pred = error_rate_for_model(model,train_docs,test_docs)
    error_rates[str(model)] = err_rate
    print('Error: \n%f %s\n' %(err_rate,model))


for model in [models_by_name['dbow+dmm'],models_by_name['dbow+dmc']]:
    print('\nEvaluating %s' %model)
    err_rate, err_count, test_count, pred = error_rate_for_model(model, train_docs, test_docs)
    error_rates[str(model)] = err_rate
    print('Error: \n%f %s\n' % (err_rate, model))


## achieved Sentiment-prediction accuracy
# compare the error rates achieved, best-to-worst
print('Err_rate Model')
for rate, name in sorted((rate,name) for name, rate in error_rates.items()):
    print("%f %s" % (rate, name))

''' 
In our testing, contrary to the result of the paper, on this problem, PV-DBOW alone performs as good as anything else.
Concatenating vectors from different models only seomtimes offers a tiny predictive improves - close to best-one  
'''

## examining result
doc_id = np.random.randint(simple_models[0].docvecs.count)
print('for doc %d.......' % doc_id)
for model in simple_models:
    inferred_docvec = model.infer_vector(alldocs[doc_id].words)
    print('%s:\n %s' % (model, model.docvecs.most_similar([inferred_docvec], topn=3)))

## Do close documents seem more related than distant ones"
import random
doc_id  = np.random.randint(simple_models[0].docvecs.count)
model = random.choice(simple_models)
sims = model.docvecs.most_similar(doc_id, topn=model.docvecs.count)
print(u'TARGET (%d): «%s»\n' % (doc_id, ' '.join(alldocs[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST',0),('MEDIAN',len(sims)//2),('LEAST',len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(alldocs[sims[index][0]].words)))

## Do the word vectors show useful similarities?
word_models = simple_models[:]
from IPython.display import HTML,display
# pick a random word with a suitable number of occurences
while True:
    words = random.choice(word_models[0].wv.index2word)
    if word_models[0].wv.vocab[words].count > 10:
        break
# or uncomment below line, to just pick a word from the relevant domain:
similars_per_model = [str(model.wv.most_similar(words,topn=20)).replace('),','),<br>\n') for model in word_models]
similar_table = ("<table><tr><th>" +
    "</th><th>".join([str(model) for model in word_models]) +
    "</th></tr><tr><td>" +
    "</td><td>".join(similars_per_model) +
    "</td></tr></table>")
print("most similar words for '%s' (%d occurences)" % (words, simple_models[0].wv.vocab[words].count))
display(HTML(similar_table))