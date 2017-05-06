import math
from collections import namedtuple, defaultdict

class CASE:
    def __init__(self, sentence, addS=0):
        self.query=[]
        if addS==1:
            if len(sentence)>0:
                self.query.append('<s>')
                for cur in sentence:
                    self.query.append(cur)
                self.query.append('</s>')
        else:
            self.query=list(sentence)
        self.length = len(self.query)
    def repl(self, cur, cand, sentence=0, char_pick=-1):
        # First::cur: the original char to change
        # Second::cand: the candidate char to change
        # Third:parameter: current sentence or original sentence
        # Fourth::char_pick: the index to pick on sentence
        if sentence==0:
            if char_pick>-1:            
                rep = list(self.query[0:char_pick+1])
            else:
                rep = list(self.query)  
            idx = self.query.index(cur)
        else:
            rep = list(sentence)
            idx = rep.index(cur)
        rep[idx]=cand
        return rep
    def candsGet(self, nlm, cande=[]):
        ncm_stats = namedtuple('ncm_prob', 'ch,prob')
        self.cands = []
        for idx,cur in enumerate(self.query):
            cur_cands = ncm.cand(cur)
            tmp =[]            
            if len(cur_cands)>=1:
                for cand in cur_cands:
                    # Put original char to first
                    if cand[0]==cur:                        
                        tmp.insert(0,ncm_stats(cand[0],cand[1]))
                    else:
                        tmp.append(ncm_stats(cand[0],cand[1]))
            else:
                tmp.append(ncm_stats(cur,1))        
            # Others
#             prob_sample = 0.0000000005            
#             if idx==0: 
#                 pre_char='' 
#             else: 
#                 pre_char = self.query[idx-1]
#             if idx==self.length-1:                
#                 post_char='' 
#             else: 
#                 post_char=self.query[idx+1]            
#             for cand in cande.cand(pre_char,post_char,1):
#                 tmp.append(ncm_stats(cand, prob_sample)) 
#             # Others
            self.cands.append(tmp)