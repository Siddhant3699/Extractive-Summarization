# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:32:57 2020

@author: Administrator
"""
import re
import numpy as np
import networkx as nx
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from topicgrouping import SentenceClustering

class ExtractiveSummarizer:
    
    def __init__(self):
        self.sw = stopwords.words('english')
        self.temp = list()
        self.summary = ""
        
    def preprocess(self,sentences):
        temp = list()
        for i in sentences:
            i = re.sub(r'\[.*?\]','',i) #remove citations
            i = re.sub(r'\(.*?\)','',i)
            i = i.replace('  ',' ')
            i = re.sub('[^a-zA-Z0-9 \n\.]','',i) #remove special characters
            temp.append(i)
        return temp
    
    def similarity(self,s1,s2):
        s1_list = word_tokenize(s1)
        s2_list = word_tokenize(s2)
        
        l1 = list()
        l2 = list()
        
        s1_set = {w for w in s1_list if not w in self.sw}
        s2_set = {w for w in s2_list if not w in self.sw}
        
        rvector = s1_set.union(s2_set)
        
        for w in rvector:
            if w in s1_set:
                l1.append(1)
            else:
                l1.append(0)
            if w in s2_set:
                l2.append(1)
            else:
                l2.append(0)
                
        c=0
        
        for i in range(len(rvector)):
            c += (l1[i]*l2[i])
        cosine_similarity = c/float((sum(l1)*sum(l2))**0.5)
        
        return cosine_similarity
    
    def build_sim_matrix(self,sentences):
        sim_matrix = np.zeros((len(sentences),len(sentences)))
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    sim_matrix[i][j] = self.similarity(sentences[i],sentences[j])
        return sim_matrix

    def textrank(self,sentences):
        self.sentences = sentences
        similarity_matrix = self.build_sim_matrix(self.sentences)
        similarity_graph = nx.from_numpy_array(similarity_matrix)
        sent_scores = nx.pagerank(similarity_graph)
        ranked_sentence = sorted(((sent_scores[i], s) for i, s in enumerate(self.sentences)), reverse=True)
        tot = 0
        for i in ranked_sentence:
            #print(i)
            tot += i[0]
        count=len(ranked_sentence)
        mean=(tot/count)
        #print(mean)
        
        #temp=list()
        for i in ranked_sentence:
            if i[0]>mean:
                #print(i[1])
                self.temp.append(i[1])
        
    def summarize(self,text):
        print(text)
        print()
        print()
        self.text = text
        self.sentences = sent_tokenize(self.text)
        self.sentences = self.preprocess(self.sentences)
        sent_cluster = SentenceClustering()
        groups = sent_cluster.group_similar_sentences(self.sentences)
        for i in groups:
            self.textrank(i)        
        for i in self.temp:
            self.summary += (" "+i)
        return self.summary.strip()
        
'''
if __name__=="__main__":
    o = ExtractiveSummarizer()
    text = "A total of 14 new flights are expected to fly out of different airports in Odisha very soon as various airlines have showed interest after Bhubaneswar airport registered a growth of 29 per cent in air traffic in the last five years, officials said. Out of the 14 new flights on cards, three flights will be operated to as many international destinations, they said. This was revealed during a high-level meeting on Enhancing Air Connectivity for Odisha on Wednesday. The air traffic growth in Bhubaneswar in the last five years is the highest at 29 per cent against the national growth rate of 17 per cent, Civil Aviation Secretary Pradeep Singh Kharola said."
    summary = o.summarize(text)
    print(summary)
'''