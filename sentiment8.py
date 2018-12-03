import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

''' Another Twitter sentiment analysis with Python — Part 8 (Dimensionality reduction: Chi2, PCA) '''
## run dimensionality reduciton on Tfidf vectors with chi-suared feature selection
'''
There are three methods you can use for feature selection with sparse matrices such tfidf vectors or count vectors.
deal with data without making it dense. 
1. chi2
    measures the lack of independence between a feature and class. 
2.mutual_info_regression
3.mutual_info_classifier
'''
## chi2