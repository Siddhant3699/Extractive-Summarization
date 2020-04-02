# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 20:40:25 2020

@author: Administrator
"""


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class SentenceClustering:
    
    def __init__(self):
        pass
    
    def group_similar_sentences(self,sentences):
        self.sentences = sentences
        
        vec = TfidfVectorizer(stop_words="english")
        
        self.X = vec.fit_transform(self.sentences)
        
        self.sil = []
        
        self.kmax = self.X.shape[0]
        
        for k in range(2,self.kmax):
            km = KMeans(n_clusters=k,init="k-means++",n_init=1,max_iter=100).fit(self.X)
            labels = km.labels_
            self.sil.append(silhouette_score(self.X, labels, metric = 'euclidean'))
            
        self.max_sil_score = -99999
        
        for i in range(len(self.sil)):
            if self.sil[i] > self.max_sil_score:
                self.max_sil_score = self.sil[i]
                
        self.optimal_k = self.sil.index(self.max_sil_score) + 2
        
        self.model = KMeans(n_clusters=self.optimal_k, init="k-means++", n_init=1, max_iter=100)
        
        self.pred = self.model.fit_predict(self.X)
        '''
        for i,j in enumerate(self.sentences):
            print(str(self.pred[i])+" : "+j)
        '''
        self.groups = list()
        
        for i in range(self.optimal_k):
            self.groups.append(list())
            
        for i in range(len(self.sentences)):
            self.groups[self.pred[i]].append(self.sentences[i])
            
        return self.groups

'''
if __name__=="__main__":
    
    o = SentenceClustering()
    sent = np.array(["This little kitty came to play when I was eating at a restaurant","What a lovely kitty he has","Google translate app in incredible","If you open 100 tabs in google you get a smiley face","Best cat photo I have ever taken","Climbing ninja cat","Impressed with google map feedback","Key promoter extension for google chrome"])
    o.group_similar_sentences(sent)
'''