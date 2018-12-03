import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
SEED = 2000

x = my_df.text
y = my_df.target
x_train, x_v_t, y_train, y_v_t = train_test_split(x, y, test_size=.02, random_state=SEED)
x_validation, x_test, y_validation, y_test = train_test_split(x_v_t,y_v_t,test_size=.5, random_state=SEED)
print("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_train),
                                                                                         (len(x_train[y_train == 0]) / (len(x_train)*1.))*100,
                                                                                         (len(x_train[y_train == 4]) / (len(x_train)*1.))*100))
print("Validation set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_validation),
                                                                                              (len(x_validation[y_validation == 0]) / (len(x_validation)*1.))*100,
                                                                                              (len(x_validation[y_validation == 4]) / (len(x_validation)*1.))*100))
print ("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_test),
                                                                                        (len(x_test[y_test == 0]) / (len(x_test)*1.))*100,
                                                                                        (len(x_test[y_test == 4]) / (len(x_test)*1.))*100))


### baseline
'''
The most popular baseline is the Zero Rule (ZeroR). ZeroR classifier simply predicts the majority category (class). 
Although there is no predictability power in ZeroR, it is useful for determining a baseline performance as a benchmark 
for other classification methods.
'''
from textblob import TextBlob
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

tbresult = [TextBlob(i).sentiment.polarity for i in x_validation]
tbpred = [0 if n < 0 else 1 for n in tbresult]
conmat = np.array(confusion_matrix(y_validation,tbpred, labels=[1,0]))
confusion = pd.DataFrame(conmat,index=['positive','negative'],
                         columns = ['predicted_positive','predicted_negative'])

print("Accuracy Score: {0:.2f}%".format(accuracy_score(y_validation, tbpred)*100))
print("-"*80)
print("Confusion Matrix\n")
print(confusion)
print("-"*80)
print("Classification Report\n")
print(classification_report(y_validation, tbpred))


## build model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from time import time

## two models to train on the different number of features, then check the accuracy of logistic regression on the
## validation set

def accuracy_summary(pipline,x_train, y_train, x_test, y_test):
    if len(x_test[y_test == 0]) / (len(x_test) * 1.) > 0.5:
        null_accuracy = len(x_test[y_test == 0]) / (len(x_test) * 1.)
    else:
        null_accuracy = 1 - len(x_test[y_test == 0]) / (len(x_test) * 1.)
    t0 = time()
    sentiment_fit = pipline.fit(x_train, y_train)
    y_pred = sentiment_fit.predict(x_test)
    train_test_time = time() - t0
    accuracy = accuracy_score(y_test, y_pred)
    print('null accuracy: {0:.2f}%'.format(null_accuracy*100))
    print('accuracy score: {0:.2f}%'.format(accuracy*100))
    if accuracy > null_accuracy:
        print('model is {0:.2f}% more accurate than null accuracy.'.format((accuracy - null_accuracy) * 100))
    elif accuracy == null_accuracy:
        print('model has the same accuracy with the null accuracy')
    else:
        print('model is {0:.2f}% less accurate than null accuracy. '.format((null_accuracy - accuracy) * 100))
    print('Train and test time: {0:.2f}s'.format(train_test_time))
    print('-'*80)
    return accuracy,train_test_time


cvec = CountVectorizer()
lr = LogisticRegression()
n_features = np.arange(10000,100001,10000)

def nfeature_accuracy_checker(vectorizer = cvec, n_features = n_features,stop_words = None, ngram_range = (1,1),classifier = lr):
    result = []
    print(classifier)
    print('\n')
    for n in n_features:
        vectorizer.set_params(stop_words = stop_words,max_features = n,ngram_range = ngram_range)
        checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        print('Validation result for {} features'.format(n))
        nfeature_accuracy, tt_time = accuracy_summary(checker_pipeline,x_train, y_train, x_validation,y_validation)
        result.append((n,nfeature_accuracy,tt_time))
    return result


## unigram
print('RESULT FOR UNIGRAM WITH STOP WORDS \n')
feature_result_wsw = nfeature_accuracy_checker(stop_words='english')
print('RESULT FOR UNIGRAM WITHOUT STOP WORDS \n')
feature_result_ug = nfeature_accuracy_checker()

'''
Check the accuracy on validation set for the different number of features by calling the 'nfeature_accuracy_checker'
I defined above. In addition, I have defined the custom stop words of top 10 frequent term to compare the result 
'''
csv = 'data/term_freq_df.csv'
term_freq_df = pd.read_csv(csv, index_col=0)
term_freq_df.sort_values(b = 'total',ascending=False).iloc[:10]

## double check if these top 10 words are actually included in SKLearn's stopword list,
from sklearn.feature_extraction import  text
a =frozenset(list(term_freq_df.sort_values(by = 'total',ascending=False).iloc[:10].index))
b = text.ENGLISH_STOP_WORDS
set(a).issubset(set(b))

my_stop_words = frozenset(list(term_freq_df.sort_values(by = 'total',ascending=False).iloc[:10].index))
print("RESULT FOR UNIGRAM WITHOUT CUSTOM STOP WORDS (Top 10 frequent words)\n")
feature_result_wocsw = nfeature_accuracy_checker(stop_words=my_stop_words)

## plot graph for three situations
nfeature_plot__wosw = pd.DataFrame(feature_result_ug, columns=['nfeatures','validation_accuracy','train_test_time'])
nfeature_plot__wocsw = pd.DataFrame(feature_result_wocsw, columns=['nfeatures','validation_accuracy','train_test_time'])
nfeature_plot__wsw = pd.DataFrame(feature_result_wsw, columns=['nfeatures','validation_accuracy','train_test_time'])
plt.figure(figsize=(8,6))
plt.plot(nfeature_plot__wosw.nfeatures, nfeature_plot__wosw.validation_accuracy, label = 'without stopwords')
plt.plot(nfeature_plot__wocsw.nfeatures, nfeature_plot__wocsw.validation_accuracy, label = 'without custom stopwords')
plt.plot(nfeature_plot__wsw.nfeatures, nfeature_plot__wsw.validation_accuracy, label = 'with stopwords')
plt.title("Without stop words VS With stop words (Unigram): Accuracy")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()
plt.show()


## Bigram
print("RESULT FOR BIGRAM WITH STOP WORDS\n")
feature_result_bg = nfeature_accuracy_checker(ngram_range=(1, 2))
print("RESULT FOR TRIGRAM WITH STOP WORDS\n")
feature_result_tg = nfeature_accuracy_checker(ngram_range=(1, 3))
nfeatures_plot_tg = pd.DataFrame(feature_result_tg,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_bg = pd.DataFrame(feature_result_bg,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_wosw = pd.DataFrame(feature_result_ug,columns=['nfeatures','validation_accuracy','train_test_time'])

plt.figure(figsize=(8,6))
plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.validation_accuracy,label='trigram')
plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy,label='bigram')
plt.plot(nfeatures_plot_wosw.nfeatures, nfeatures_plot_wosw.validation_accuracy, label='unigram')
plt.title("N-gram(1~3) test result : Accuracy")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()


### Take a closer look at best performing number of features with each n-gram. Below function not only reports accuracy
### but also gives confusion matrix and clssification report
def train_test_and_evaluate(pipeline, x_train, y_train, x_test, y_test):
    if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.5:
        null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
    else:
        null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
    sentiment_fit = pipeline.fit(x_train, y_train)
    y_pred = sentiment_fit.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    conmat = np.array(confusion_matrix(y_test, y_pred, labels=[0,1]))
    confusion = pd.DataFrame(conmat, index=['negative', 'positive'],
                         columns=['predicted_negative','predicted_positive'])
    print("null accuracy: {0:.2f}%".format(null_accuracy*100))
    print("accuracy score: {0:.2f}%".format(accuracy*100))
    if accuracy > null_accuracy:
        print("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100))
    elif accuracy == null_accuracy:
        print("model has the same accuracy with the null accuracy")
    else:
        print("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100))
    print("-"*80)
    print("Confusion Matrix\n")
    print(confusion)
    print("-"*80)
    print("Classification Report\n")
    print(classification_report(y_test, y_pred, target_names=['negative','positive']))


tg_cvec = CountVectorizer(max_features=80000,ngram_range=(1, 3))
tg_pipeline = Pipeline([
        ('vectorizer', tg_cvec),
        ('classifier', lr)
    ])
train_test_and_evaluate(tg_pipeline, x_train, y_train, x_validation, y_validation)