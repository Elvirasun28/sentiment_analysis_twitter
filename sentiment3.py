import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

term_freq_df = pd.read_csv('data/term_freq_df.csv', encoding='utf-8',index_col= 0)
''' Another Twitter sentiment analysis with Python — Part 3 (Zipf’s Law, data visualisation)  '''
y_pos = np.arange(500)
plt.figure(figsize=(10,8))
s = 1
expected_zipf = [term_freq_df.sort_values(by = 'total',ascending = False)['total'][0] / (i + 1) ** s for i in y_pos]
plt.bar(y_pos, term_freq_df.sort_values(by = 'total', ascending= False)['total'][:500], align = 'center',alpha = 0.5)
plt.plot(y_pos, expected_zipf,color = 'r',linestyle = '--', linewidth = 2, alpha = 0.5)
plt.ylabel('Frequency')
plt.title('Top 500 tokens in tweets')


### Another way to plot this is on a log-log graph, with X-axis being log(rank), Y-axis being log(freq)

from pylab import *
counts = term_freq_df.total
tokens = term_freq_df.index
ranks = np.arange(1, len(counts) + 1)
indices = np.argsort(-counts)
frequencies = counts[indices]
plt.figure(figsize=(8,6))
plt.xlim(1,10**6)
plt.ylim(1,10**6)
plt.loglog(ranks, frequencies,marker = '.')
plt.plot([1,frequencies[0]],[frequencies[0],1],color='r')
plt.title("Zipf plot for tweets tokens")
plt.xlabel("Frequency rank of token")
plt.ylabel("Absolute frequency of token")
plt.grid(True)
for n in list(np.logspace(-0.5, np.log10(len(counts)-2), 25).astype(int)):
    dummy = plt.text(ranks[n], frequencies[n], " " + tokens[indices[n]],
                 verticalalignment="bottom",
                 horizontalalignment="left")


### the stop words will not help much, because of the same high-freq words, such as 'the','to', will equally
### frequent in both class. If these stop words dominate both of the classes, I won't be able to have a meaningful
### result. So I  decide to remove stop words, and also will limit the max_features to 10000 with countervectorizer
from sklearn.feature_extraction.text import  CountVectorizer
cvec = CountVectorizer(stop_words='english',max_features=10000)
cvec.fit(my_df.text)
document_matrix = cvec.transform(my_df.text)
## negative part
neg_batches = np.linspace(0,798179,10).astype(int)
i = 0
neg_tf = []
while i < (len(neg_batches) - 1):
    batch_result = np.squeeze(np.asarray(np.sum(document_matrix[neg_batches[i]:neg_batches[i+1]],axis=0)))
    neg_tf.append(batch_result)
    print(neg_batches[i+1],"entries' term frequency calculated")
    i += 1

## positive part
pos_batches = np.linspace(798179,1596019,10).astype(int)
i = 0
pos_tf = []
while i < (len(pos_batches) - 1):
    batch_result = np.squeeze(np.asarray(np.sum(document_matrix[pos_batches[i]:pos_batches[i+1]],axis=0)))
    pos_tf.append(batch_result)
    print(pos_batches[i+1],"entries' term frequency calculated")
    i += 1

neg = np.sum(neg_tf, axis = 0)
pos = np.sum(pos_tf, axis = 0)
term_freq_df2 = pd.DataFrame([neg,pos],columns=cvec.get_feature_names()).transpose()
term_freq_df2.columns = ['negative', 'positive']
term_freq_df2['total'] = term_freq_df2['negative'] + term_freq_df2['positive']
term_freq_df2.sort_values(by='total', ascending=False).iloc[:10]

### Let's see what are the top 50 words in negative tweets on a bar chart
y_pos = np.arange(50)
plt.figure(figsize=(12,10))
plt.bar(y_pos, term_freq_df2.sort_values(by ='negative',ascending=False)['negative'][:50],align = 'center',alpha = 0.5)
plt.xticks(y_pos, term_freq_df2.sort_values(by = 'negative',ascending=False)['negative'][:50].index, rotation = 'vertical')
plt.ylabel('Frequency')
plt.xlabel('Top 50 negative tokens')
plt.title('Top 50 tokens in negative tweets')

### Let's see wjat are tje top 50 words in positive tweets on a bar chart
y_pos = np.arange(50)
plt.figure(figsize=(12,10))
plt.bar(y_pos, term_freq_df2.sort_values(by = 'positive',ascending=False)['positive'][:50],align = 'center',alpha = 0.5)
plt.xticks(y_pos, term_freq_df2.sort_values(by = 'positive', ascending=False)['positive'][:50].index, rotation = 'vertical')
plt.ylabel('Frequency')
plt.xlabel('Top 50 positive tokens')
plt.title('Top 50 tokens in positive tweets')

### plot the negative frequency of a word on X-axis and the positive frequency on Y-axis
import seaborn as sns
plt.figure(figsize=(8,6))
ax = sns.regplot(x = 'negative',y='positive',fit_reg = False, scatter_kws={'alpha' : 0.5},data = term_freq_df2)
plt.ylabel('Positive Freq')
plt.xlabel('Negative Freq')
plt.title('Negative Frequency vs Positive Frequency')
plt.show()
# most of the words are below 10000 on both x-axis and y-axis, and we cannot see meaningful relations between negative
# and positive frequency


### if a word appears more often in one class compared to another, this can be a good measure of how much the word is
### meaningful to characterise the class
term_freq_df2['pos_rate'] = term_freq_df2['positive'] * 1./term_freq_df2['total']
term_freq_df2.sort_values(by = 'pos_rate', ascending=False).iloc[:10]

### words with highest pos_rate have zero freq in the negative tweets, but overall freq of these words are too low to
### think of it as a guidline for positive tweets
term_freq_df2['pos_freq_pct'] = term_freq_df2['positive'] * 1./term_freq_df2['positive'].sum()
term_freq_df2.sort_values(by = 'pos_freq_pct', ascending= False).iloc[:10]
# since pos_freq_pct is just the frequency scaled over the total sum of the frequency, the rank of pos_freq_pct is
# exactly same as just the positive frequency.

### combine pos_rate, pos_freq_pct -> harmonic mean
### mitigate the impact of large outliers and aggravate the impact of small ones
from scipy.stats import hmean
term_freq_df2['pos_hmean'] = term_freq_df2.apply(lambda x: (hmean([x['pos_rate'], x['pos_freq_pct']])
                                                            if x['pos_rate'] > 0 and x['pos_freq_pct'] > 0 else 0), axis=1)
term_freq_df2.sort_values(by = 'pos_hmean',ascending=False).iloc[:10] # the same as pos_freq_pct

### cumulative distribution function
from scipy.stats import norm
def normcdf(x):
    return norm.cdf(x, x.mean(),x.std())

term_freq_df2['pos_rate_normcdf'] = normcdf(term_freq_df2['pos_rate'])
term_freq_df2['pos_freq_pct_normcdf'] = normcdf(term_freq_df2['pos_freq_pct'])
term_freq_df2['pos_normcdf_hmean'] = hmean([term_freq_df2['pos_rate_normcdf'],term_freq_df2['pos_freq_pct_normcdf']])

term_freq_df2.sort_values(by = 'pos_normcdf_hmean',ascending=False).iloc[:10]

### same to negative part
term_freq_df2['neg_rate'] = term_freq_df2['negative'] * 1./term_freq_df2['total']
term_freq_df2['neg_freq_pct'] = term_freq_df2['negative'] * 1./term_freq_df2['negative'].sum()
term_freq_df2['neg_rate_normcdf'] = normcdf(term_freq_df2['neg_rate'])
term_freq_df2['neg_freq_pct_normcdf'] = normcdf(term_freq_df2['neg_freq_pct'])
term_freq_df2['neg_normcdf_hmean'] = hmean([term_freq_df2['neg_rate_normcdf'],term_freq_df2['neg_freq_pct_normcdf']])
term_freq_df2.sort_values(by = 'neg_normcdf_hmean', ascending=False).iloc[:10]

plt.figure(figsize=(8,6))
ax = sns.regplot(x = 'neg_normcdf_hmean',y = 'pos_normcdf_hmean', fit_reg=False, scatter_kws={'alpha':0.5}, data=term_freq_df2)
plt.ylabel('Positive Rate and Frequency CDF Harmonic Mean')
plt.xlabel('Negative Rate and Frequency CDF Harmonic Mean')
plt.title('neg_normcdf_mean vs pos_normcdf_mean')

### alternative method to plot normcdf_hmean
from bokeh.plotting import figure
from bokeh.io import output_notebook,show
from bokeh.models import LinearColorMapper
from bokeh.models import HoverTool

output_notebook()
color_mapper = LinearColorMapper(palette='Inferno256',
                                 low = min(term_freq_df2.pos_normcdf_hmean),
                                 high = max(term_freq_df2.neg_normcdf_hmean))
p = figure(x_axis_label = 'neg_normcdf_hmean', y_axis_label = 'pos_normcdf_hmean')
p.circle('neg_normcdf_hmean','pos_normcdf_hmean',size=5,alpha=0.3,source=term_freq_df2,color={'field': 'pos_normcdf_hmean', 'transform': color_mapper})
hover = HoverTool(tooltips=[('token','@index')])
p.add_tools(hover)
show(p)
