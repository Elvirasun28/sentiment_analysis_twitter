import pandas as pd
import numpy as np
import chardet
import matplotlib.pyplot as plt


''' Another Twitter sentiment analysis with Python — Part 1 '''
# columns name
cols = ['sentiment','id','date','query_string','user','text']
df = pd.read_csv('data/training.1600000.processed.noemoticon.csv',encoding='Latin-1',sep = ',',header=None)
df.columns = cols
df.sentiment.value_counts()
# drop the unneeded cols
df.drop(['id','date','query_string','user'], axis = 1, inplace=True)

## All the negative class is from 0 -799999th index and the positive class entries start from 800000 to the end of
## dataset

### data preparation
df['pre_clean_len'] = [len(t) for t in df.text]
# data dictionary
from pprint import pprint
data_dict = {
    'sentiment':{
        'type': df.sentiment.dtype,
        'description': 'sentiment class - 0: negative, 1: positive'
    },
    'text':{
        'type': df.text.dtype,
        'description': 'tweet text'
    },
    'pre_clean_len':{
        'type':df.pre_clean_len.dtype,
        'description': 'Length of the tweet before cleaning'
    },
    'dataset_shape': df.shape
}
pprint(data_dict)
# plot pre_clean_len with box plot and see the overall distribution of length of strings in each entry
plt.figure()
plt.boxplot(df.pre_clean_len)
plt.show()
## this looks a bit strange, since the twitters' char limit is 140. But from the above plot, some of the tweets are way
## more than 140 characters 1ong
df[df.pre_clean_len > 140]

'''
Data Preparation steps: 
1.Souping
2.BOM removing
3.url address(‘http:’pattern), twitter ID removing
4.url address(‘www.'pattern) removing
5.lower-case
6.negation handling
7.removing numbers and special characters
8.tokenizing and joining
'''
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
import re

tok = WordPunctTokenizer()

pat1 = r'@[A-Za-z0-9_]+'
pat2 = r'https?://[^ ]+'
combined_pat = r'|'.join((pat1, pat2))
www_pat = r'www.[^ ]+'
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

def tweet_cleaner_updated(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_removed = souped
    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
    lower_case = stripped.lower()
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
    return (" ".join(words)).strip()

# clean the dataset into 4 parts
nums = [0,400000,800000,1200000,1600000]
print('cleaning and parsing the tweets ....\n')
clean_tweet_texts = []
def process_clean_data(num1,num2):
    for i in range(num1,num2):
        if (i+1) % 10000 == 0 :
            print('Tweets %d of %d has been processed...' % (i + 1, num2))
        clean_tweet_texts.append(tweet_cleaner_updated(df['text'][i]))
    return True

for i in range(len(nums)):
    process_clean_data(nums[i],nums[i+1])

### save cleaned data as csv
clean_df = pd.DataFrame(clean_tweet_texts,columns=['text'])
clean_df['target'] = df.sentiment
clean_df.to_csv('data/clean_tweet.csv',encoding='utf-8')
