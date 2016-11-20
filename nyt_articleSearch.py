# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 15:05:38 2016

@author: adam
"""
from nytAPI import articleAPI
import pandas as pd

def getArticles():
    allArticles = []
    for i in range(0,100):
        print(i)
        try:
            articles = api.search(q = 'Trump', 
                              fq={'source':['Reuters','AP', 'The New York Times']},
                              page = str(i),
                              begin_date = 20000101,
                              end_date = 20161230)
            allArticles = allArticles + articles['response']['docs']
        except Exception:
            pass
            
    return pd.DataFrame(allArticles)


if __name__ == '__main__':
    api = articleAPI('YOUR API KEY')
                            
    df = getArticles()
    
    
    articleHeadlines = []
    for item in df.headline:
        try:
            articleHeadlines.append(item['print_headline'])
        except:
            articleHeadlines.append(item['main'])
    
    
