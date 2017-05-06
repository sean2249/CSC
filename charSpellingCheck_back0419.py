
# coding: utf-8

# In[1]:

# from model.case import CASE
from model.lm import LM
from model.ncm import NCM

import os 
import pickle
import math
from collections import namedtuple, defaultdict
import re

import time 
import sys, getopt


# In[2]:

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
    def candsGet(self, ncm, candsExpandTag=0):
        ncm_stats = namedtuple('ncm_prob', 'ch,prob')
        self.cands = []
        for idx,cur in enumerate(self.query):
            cur_cands = ncm.cand(cur)
            tmp =[]          
            # Have more than one candidatae (Except me)
            if len(cur_cands)>=1:
                for cand in cur_cands:
                    # Put original char to first
                    if cand[0]==cur:                        
                        tmp.insert(0,ncm_stats(cand[0],cand[1]))
                    else:
                        tmp.append(ncm_stats(cand[0],cand[1]))
            else:
                tmp.append(ncm_stats(cur,1))        
            # ========  
            # Others
            # ========  
            if candsExpandTag != 0:
                prob_sample = 0.0000000005            
                if idx==0: 
                    pre_char='' 
                else: 
                    pre_char = self.query[idx-1]
                if idx==self.length-1:                
                    post_char='' 
                else: 
                    post_char=self.query[idx+1]            
                for cand in cande.cand(pre_char,post_char,1):
                    tmp.append(ncm_stats(cand, prob_sample)) 
            # ========  
            # Others
            # ========  
            
            self.cands.append(tmp)


# In[3]:

data_root = '/home/kiwi/udn_data/training_confusion/'
#  data_root = 'G:/UDN/training_confusion/'
channel_filename = data_root+'channelModel.pkl'
lm_filename = data_root+'sinica.corpus.seg.char.lm'
# lm_filename = data_root+'trigram_2.lm'
# lm_filename = '/home/kiwi/Documents/udn_data/trigram.lm'
ncmEx_filename = data_root+'dict_word.txt'

lm = LM(lm_filename)
ncm = NCM(channel_filename)
# candsExpand = CANDSEXPAND(ncmEx_filename)


# In[4]:

def viterbi_ng(seq, par={}, show=0):
    # LM,NCM required
    para = par.get('weight',[0.7,0.3])
    ng_num = par.get('ngnum', 3)
#     candsExpandTag = par['candsExpand']
    
    threshold = 0.05
    
    case = CASE(seq, 1)
    case.candsGet(ncm)    
    
    sect_named = namedtuple('sect_stat', 'score, length, idx, seq')
    
    cands4all =[]
    pre_cand_lst = []
    
    for cur_idx, cur_ch in enumerate(case.query):  
        section = []
        if show==1:
            print('==========')
            print(cur_idx, cur_ch, ' '.join([x.ch for x in case.cands[cur_idx]]))
        

        if cur_idx==0 and cur_ch=='<s>':
            section.append({'from_to':'<s>', 'score':0.0, 'seq':['<s>'], 'idx_seq':[0]})

        elif cur_idx>0:                
            for cand_idx, cand in enumerate(case.cands[cur_idx]):
                sect_ncm = math.log10(cand.prob)              
            
                batch = []
                
                # lm.scoring(list)
                for pre_idx, pre_cand in enumerate(pre_cand_lst):
                    batch_seq = pre_cand['seq'] + [cand.ch]
                    
                    # Compute before?
                    batch_lm = lm.scoring(batch_seq[-ng_num:], ng_num) + pre_cand['score']
                    batch_score = (para[0]*batch_lm+para[1]*sect_ncm)
                    
                    if show==1: 
                        print('Seq:%s\tpre:%.2f\tbLM:%.2f\tNCM:%.2f\tTotal:%.2f'                               %(batch_seq[-ng_num:], pre_cand['score'], batch_lm, sect_ncm, batch_score))
                    
                    ttt = list(pre_cand['idx_seq'])                    
                    ttt.append(cand_idx)
                
                    batch_dict = {'from_to':batch_seq[-ng_num:], 'score':batch_score,                                   'seq':batch_seq, 'idx_seq':ttt}
                    
                    batch.append(batch_dict)               
                
                best = max(batch, key=lambda x:x['score'])
                if show==1: 
                    print('== Choose Seq:%s\t NCM:%.2f\tTotal:%.2f'                         %(best['from_to'], sect_ncm, best['score']))
                                
                section.append(best)
                
        pre_cand_lst = list(section)
        cands4all.append(pre_cand_lst)
    
    winner = max(pre_cand_lst, key=lambda x:x['score'])
    output = ''.join(winner['seq'][1:-1])
        
    result = dict(orig=seq, correct=output, log=cands4all, win=winner, cor_idx=winner['idx_seq'][1:-1])    
    return result


# In[5]:

def run_test(testfilename, resultname, par):    
    show = par.get('show',0)
    
    with open(testfilename, 'r',encoding='utf8') as fp, open(resultname,'w',encoding='utf8') as wp:
        for line in fp:        
            line = line.strip('\n')
            idx1 = line.find('=')+1
            idx2 = line.find(')')
            dataNum = line[idx1:idx2]
            seq = line[idx2+2:]        

            if show==1: print('=====')
            print(dataNum)   
            
            errors = batch(seq, par)
            
            wp.write(dataNum)
            if len(errors)!=0:
                for error in errors:
                    wp.write(', ')
                    wp.write(', '.join(error))
            else:
                wp.write('0')
                
            wp.write('\n')
                


# In[6]:

def main(argv, par):
    if len(argv) < 3:
        print('Usage: python filename.py token test_file1 test_file2')
        sys.exit(1)
    else:
        data_length = len(argv)-2
        test_data = []

        token = argv[1]

        for idx, item in enumerate(argv[2:],1):
            test_data_batch = item
            print('%d/%d- loading %s...'                    %(idx, data_length, test_data_batch))
            if not(os.path.exists(test_data_batch)):
                print('Unable to load file')
            else:
                test_data.append(test_data_batch)
              
    for idx, batch_name in enumerate(test_data, 1):
        print('=========')
        print('Batch %d-%s' %(idx, batch_name))
        print('=========')
        result_name = 're_%s_%d_new.txt' %(str(token), idx)
        run_test(batch_name, result_name, par)


# In[12]:

def debug_ncm(ch, append=None, value=0.05, show=0):    
    tt = ncm.cand(ch)
    if show==1:
        for d in tt:
            print(d)
        
    if append:
        ncm.table[ch][append] = value
        if show==1:
            print('== Add %s to Set of %s' %(append,ch))
    else:
        return tt


# In[8]:

def debug_lm(seq, ngnum=2):
    lst = seq.split()
    for item in lst:
        print(item, lm.scoring(item, ngnum))


# In[9]:

def seperateSeq(seq):
    pattern = re.compile('[，。！]')
    
    pre_idx=0
    output = []
    for idx, ch in enumerate(seq):
        if pattern.search(ch):
            tmp = seq[pre_idx:idx+1]
            output.append(tmp)
            pre_idx = idx+1
    
    if pre_idx<len(seq):
        tmp = seq[pre_idx:idx+1]
        output.append(tmp)
        
    return output 


# In[ ]:

def nnn(seq, show):
    
    case = CASE(seq, 1)  
    case.candsGet(ncm)    

    for cur_idx, cur_ch in enumerate(case.query):
        
        if show==1:
            print('==========')
            print(cur_idx, cur_ch, ' '.join([x.ch for x in case.cands[cur_idx]]))
            
        
    
    '''
    Past 
    '''
        
    sect_named = namedtuple('sect_stat', 'score, length, idx, seq')
    
    cands4all =[]
    pre_cand_lst = []
    
    for cur_idx, cur_ch in enumerate(case.query):  
        section = []
        if show==1:
            print('==========')
            print(cur_idx, cur_ch, ' '.join([x.ch for x in case.cands[cur_idx]]))
        

        if cur_idx==0 and cur_ch=='<s>':
            section.append({'from_to':'<s>', 'score':0.0, 'seq':['<s>'], 'idx_seq':[0]})

        elif cur_idx>0:                
            for cand_idx, cand in enumerate(case.cands[cur_idx]):
                sect_ncm = math.log10(cand.prob)              
            
                batch = []
                
                # lm.scoring(list)
                for pre_idx, pre_cand in enumerate(pre_cand_lst):
                    batch_seq = pre_cand['seq'] + [cand.ch]
                    
                    # Compute before?
                    batch_lm = lm.scoring(batch_seq[-ng_num:], ng_num) + pre_cand['score']
                    batch_score = (para[0]*batch_lm+para[1]*sect_ncm)
                    
                    if show==1: 
                        print('Seq:%s\tpre:%.2f\tbLM:%.2f\tNCM:%.2f\tTotal:%.2f'                               %(batch_seq[-ng_num:], pre_cand['score'], batch_lm, sect_ncm, batch_score))
                    
                    ttt = list(pre_cand['idx_seq'])                    
                    ttt.append(cand_idx)
                
                    batch_dict = {'from_to':batch_seq[-ng_num:], 'score':batch_score,                                   'seq':batch_seq, 'idx_seq':ttt}
                    
                    batch.append(batch_dict)               
                
                best = max(batch, key=lambda x:x['score'])
                if show==1: 
                    print('== Choose Seq:%s\t NCM:%.2f\tTotal:%.2f'                         %(best['from_to'], sect_ncm, best['score']))
                                
                section.append(best)
                
        pre_cand_lst = list(section)
        cands4all.append(pre_cand_lst)
    
    winner = max(pre_cand_lst, key=lambda x:x['score'])
    output = ''.join(winner['seq'][1:-1])
        
    result = dict(orig=seq, correct=output, log=cands4all, win=winner, cor_idx=winner['idx_seq'][1:-1])    
    return result


# In[10]:

def batch(seq, par = dict(weight=[0.7,0.3],ngnum=3), show=0):            
    sub_seqs = seperateSeq(seq)
    
    total_length = 0
    error = []
    for sub in sub_seqs:
        if show==1: 
            print(sub, len(sub))
        
        result = viterbi_ng(sub, par, show)
        
        tmp_pos   = [i for i,e in enumerate(result['cor_idx']) if e!=0]
        pos_error = [str(idx+total_length+1) for idx in tmp_pos]     
        ch_error  = [result['correct'][int(idx)] for idx in tmp_pos]
        
        if show==1:
            for idx in tmp_pos:
                print(result['correct'][int(idx)])
            print(result['orig'])
            print(result['correct'])            
        
        total_length += len(sub)
        for pos, ch in zip(pos_error, ch_error):
            error.append((pos, ch))
            
    return error


# In[ ]:

batch('')


# In[21]:

if __name__=='__main__':
    par = dict(weight=[0.7,0.3],ngnum=3)
    main(sys.argv, par)
    
    

