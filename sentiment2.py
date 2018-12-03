import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

my_df = pd.read_csv('data\clean_tweet.csv',index_col= 0)

''' Reclean the data and drop the null text row '''
my_df.info()
my_df[my_df.isnull().any(axis = 1)].head()
np.sum(my_df.isnull().any(axis = 1))
# check the original text for those null value texts
df = pd.read_csv('data/training.1600000.processed.noemoticon.csv',header = None, encoding='Latin-1')
df.iloc[my_df[my_df.isnull().any(axis = 1)].index,:].head()
''' 
It seems like only text information they had was either twitter ID or url address. Anyway, these are the info
I decide to discard for the sentiment analysis, so i will drop these null rows, and update the data frame 
'''
my_df.dropna(inplace=True)
my_df.reset_index(drop=True,inplace=True)
my_df.info()

### word cloud
from wordcloud import WordCloud
def wordcloud_plot(sentiment_pos):
    tweets = my_df[my_df.target == sentiment_pos]
    string = []
    for t in tweets.text:
        string.append(t)
    string = pd.Series(string).str.cat(sep = ' ')

    wordcloud = WordCloud(width=1600, height=800, max_font_size=200,colormap='magma').generate(string)
    plt.figure(figsize=(12,10))
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.axes('off')
    plt.show()

wordcloud_plot(sentiment_pos = 0) # plot the negative review
wordcloud_plot(sentiment_pos = 4) # plot the positive review

'''
Note: 
1. even though the tweets contain the word “love”, in these cases it is negative sentiment, 
because the tweet has mixed emotions like “love” but “miss”. Or sometimes used in a 
sarcastic way.
2. the word “work” was quite big in negative word cloud, but also quite big in positive word 
cloud. It might implies that many people express negative sentiment towards work, but also many 
people are positive about works.

'''

## term frequencies
## include removing stop words, limiting the maximum number of terms
### CountVectorizer can lowercase letters, disregard punctuation and stopwords, but it can't LEMMATIZE or STEM
from sklearn.feature_extraction.text import  CountVectorizer
cvec = CountVectorizer()
cvec.fit(my_df.text)
len(cvec.get_feature_names())

neg_doc_matrix = cvec.transform(my_df[my_df.target == 0].text)
pos_doc_matrix = cvec.transform(my_df[my_df.target == 4].text)
neg_tf = np.sum(neg_doc_matrix, axis = 0)
pos_tf = np.sum(pos_doc_matrix, axis = 0)
neg = np.squeeze(np.asarray(neg_tf))
pos = np.squeeze(np.asarray(pos_tf))
term_freq_df = pd.DataFrame([neg,pos],columns=cvec.get_feature_names()).transpose()
term_freq_df = term_freq_df.iloc[term_freq_df.index.get_loc('aa'):]
term_freq_df.columns = ['negative','positive']
term_freq_df['total'] = term_freq_df['negative'] + term_freq_df['positive']
term_freq_df.sort_values(by='total', ascending=False).iloc[:10]
term_freq_df.to_csv('data/term_freq_df.csv',encoding='utf-8')