'''
ANN with Tfidf vectorizer
The best performing Tfidf vectors I got is with 100,00 features including up to trigram with logistic regression.
Validation accuracy is 82.91%, while train set accuracy is 84.19%.
I would want to see if the neural netowrk can boost the performance of my existing tfidf vectors
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
seed = 7
np.random.seed(seed)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

'''
The structure of below NN model has 100,000 nodes in the input layer, then 64 nodes in a hidden layer with Relu
activation function applied, then finally one output layer with sigmoid activation function applied. 
- ADAM optimizing  (combines two methods of optimisation: RMSProp, Momentum. )
- binary cross entropy loss 
- Keras NN model cannot handle sparse matrix directly. The data has to be dense array or matrix, but 
transforming the whole training data Tfidf vectors of 1.5 million to dense array won’t fit into my RAM. 
So I had to define a function, which generates iterable generator object, so that it can be fed to NN model.
(
Note that the output should be a generator class object rather than directly returning arrays, this can be achieved 
by using “yield” instead of “return”.
)
'''
csv = 'data/clean_tweet.csv'
my_df = pd.read_csv(csv,index_col=0)
my_df.dropna(inplace=True)
my_df.reset_index(drop=True,inplace=True)
## get a sample for studying only
my_df = pd.concat([my_df[my_df.target == 0][:int(len(my_df)*0.1)],my_df[my_df.target == 4][:int(len(my_df)*0.1)]])
my_df.info()

x= my_df.text
y=my_df.target

## separate the dataset to train & test dataset
SEED = 2000
x_train, x_validation_test, y_train, y_validation_test = train_test_split(x,y,test_size=.02, random_state=SEED)
x_validation, x_test, y_validation, y_test = train_test_split(x_validation_test,y_validation_test,test_size=.5, random_state=SEED)

print("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_train),
                                                                             (len(x_train[y_train == 0]) / (len(x_train)*1.))*100,
                                                                            (len(x_train[y_train == 1]) / (len(x_train)*1.))*100))
print("Validation set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_validation),
                                                                             (len(x_validation[y_validation == 0]) / (len(x_validation)*1.))*100,
                                                                            (len(x_validation[y_validation == 1]) / (len(x_validation)*1.))*100))
print("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_test),
                                                                             (len(x_test[y_test == 0]) / (len(x_test)*1.))*100,
                                                                            (len(x_test[y_test == 1]) / (len(x_test)*1.))*100))


## ANN with Tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
tvec1 = TfidfVectorizer(max_features=100000,ngram_range=(1,3))
tvec1.fit(x_train)

x_train_tfidf = tvec1.transform(x_train)
x_validation_tfidf = tvec1.transform(x_validation).toarray()

clf = LogisticRegression()
clf.fit(x_train_tfidf,y_train)
clf.score(x_validation_tfidf,y_validation)
clf.score(x_train_tfidf,y_train)


print('ANN Building Begins..............................')
def batch_generator(X_data, y_data, batch_size):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    index = np.arange(np.shape(y_data)[0])
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X_data[index_batch,:].toarray()
        y_batch = y_data[y_data.index[index_batch]]
        counter += 1
        yield X_batch,y_batch
        if (counter > number_of_batches):
            counter=0

model = Sequential()
model.add(Dense(64, activation='relu',input_dim = 100000))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit_generator(generator=batch_generator(x_train_tfidf,y_train,32),
                    epochs=5,validation_data=(x_validation_tfidf,y_validation),
                    steps_per_epoch=x_train_tfidf.shape[0]/32)

from sklearn.preprocessing import Normalizer
norm = Normalizer().fit(x_train_tfidf)
x_train_tfidf_norm = norm.transform(x_train_tfidf)
x_validation_tfidf_norm = norm.transform(x_validation_tfidf)

model_n = Sequential()
model_n.add(Dense(64,activation='relu',input_dim=1000000))
model_n.add(Dense(1,activation='sigmoid'))
model_n.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

'''
By the look of the result, normalizing seems to have almost no effect on the performance. And it is at this point 
I realized that Tfidf is already normalized by the way it is calculated. TF (Term Frequency) in Tfidf is not absolute 
frequency but relative frequency, and by multiplying IDF (Inverse Document Frequency) to the relative term frequency 
value, it further normalizes the value in a cross-document manner
'''

model1 = Sequential()
model1.add(Dense(64, activation='relu', input_dim=100000))
model1.add(Dropout(0.2))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model1.fit_generator(generator=batch_generator(x_train_tfidf, y_train, 32),
                    epochs=5, validation_data=(x_validation_tfidf, y_validation),
                    steps_per_epoch=x_train_tfidf.shape[0]/32)