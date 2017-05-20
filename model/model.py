
# coding: utf-8

# In[1]:

from collections import namedtuple
import pickle
# import sys
# import math
# import re
# from nltk import bigrams,trigrams


# In[ ]:

class LM:
    def __init__(self, lm_filename):
        print('Loading language model %s ...' %(lm_filename))
        ngram_stats = namedtuple('lm_prob', 'p,bp')
        self.table = {}
        with open(lm_filename, 'r', encoding='utf8') as fp:
            for line in fp:
                seq = line.strip().split('\t')
                if len(seq)>=2:
                    (word, prob, backprob) = (tuple(seq[1].split()), float(seq[0]), 
                                        float(seq[2] if len(seq)==3 else 0.0))
                    if len(word) > 3:
                        break
                    self.table[word] = ngram_stats(prob, backprob)
    def begin(self,state):
        return ('<s>', state)
    
    def end(self,state):
        return (state, '</s>')
    
    def score_batch(self, seq):
        ngram_stats = namedtuple('lm_prob', 'p,bp')
        failed = ngram_stats(0, 'NotFound')

        score = 0.0             
        while len(seq)>0:
            if seq in self.table:
                return score + self.table[seq].p
            else:                                
                back_prob = self.table.get(seq[:-1], failed).bp
                if back_prob == 'NotFound':
                    if len(seq)==1:
                        back_prob = -99
                    else:
                        back_prob = self.score_batch(seq[:-1])
                score += back_prob                
                seq = seq[1:]
        
        return score        
    
    def scoring(self, seq, ngnum=2, show=0):
        # sentence msut be list                
        score = 0.0
                
        pairs = []
        for idx in range(len(seq)):
            if idx==0 and seq[idx]=='<s>':
                continue
            seq_idx = 0 if idx<ngnum else idx-ngnum+1
            pairs.append(tuple(seq[seq_idx:idx+1]))                           
            
        for pair in pairs:
            score += self.score_batch(pair)            
            if show==1:
                print(pair, ":\t", score)
        
        if show == 1:          
            print('========')
            print(seq, ":\t", score)
                
        return score                        


# In[96]:

class NCM:
    def __init__(self, channel_filename, ncm_global=[]):
        print('Loading channel model %s ...' %(channel_filename))
        with open(channel_filename, 'rb') as fp:
            self.table = pickle.load(fp, encoding='utf8')
        self.ncm_global = ncm_global
                        
    def get_cands(self, cur_char):       
        cand_dict = dict(self.table.get(cur_char,[]))
        cur_prob = 1.0 if not cand_dict else cand_dict.pop(cur_char)
        
        if not self.ncm_global:        
            query_cands = [(cur_char,cur_prob)] 
            query_cands.extend(cand_dict.items())                
        else:            
            base = self.ncm_global * (1-len(cand_dict)) + len(cand_dict)
            query_cands = [(cur_char, self.ncm_global/base)]
            query_cands.extend((c,(1-self.ncm_global)/base) for c,p in cand_dict.items())
   
        return query_cands


# In[ ]:

class CASE:
    def __init__(self, sentence, ncm):
        assert type(sentence) == str, 'Input must be string'
        assert len(sentence) > 0, 'Input must have content'
        self.query=[]
        self.query.append('<s>')
        self.query.extend(list(sentence))
        self.query.append('<s>')
            
        # get candidate
        self.cands = []
        for cur_ch in self.query:
            self.cands.append(ncm.get_cands(cur_ch))


# In[ ]:

if __name__=='__main__':
    lm = LM('../sinica.corpus.seg.char.lm')
#     lm2 = LM2('../sinica.corpus.seg.char.lm')

    ncm_filename = 'G:/UDN/training_confusion/channelModel.pkl'

