
# coding: utf-8

# In[15]:

# from model.case import CASE
from model.model import LM, CASE, NCM


import os 
import pickle
import math
from collections import defaultdict
import re
import pandas as pd 

import time 
import sys, getopt


# In[2]:

# data_root = '/home/kiwi/udn_data/training_confusion/'
data_root = 'G:/UDN/training_confusion/'
channel_filename = data_root+'channelModel.pkl'
lm_filename = data_root+'sinica.corpus.seg.char.lm'
# lm_filename = data_root+'trigram_2.lm'
# lm_filename = '/home/kiwi/Documents/udn_data/trigram.lm'
ncmEx_filename = data_root+'dict_word.txt'
con_filename = './extractUDN_prepost/all.csv'

# candsExpand = CANDSEXPAND(ncmEx_filename)


# In[3]:

class CONFUSION:
    def __init__(self, filename, con_log_file):
        print('Loading Preprocess_RuleBased model {} ...'.format(filename))
        self.df = pd.DataFrame.from_csv(con_filename)
        if os.path.exists(con_log_file):
            os.remove(con_log_file)
    
    def organize(self, label=[], threshold=10):
        if set(label).issubset(set(self.df)) and len(label)>0:
            self.ptable = self.df.groupby(label)
        else:
            label = ['pre','error','post','corr']
            self.ptable = self.df.groupby(label)
            
        self.ptable = self.speDataframe(self.ptable)
        self.ptable = self.ptable.loc[self.ptable['count']>threshold]
        self.label_len = len(label)-1
        
    def speDataframe(self,_gg):
        _ggS = _gg.size()
        _ggDF = pd.DataFrame(_ggS,columns=['count'])
        _ggDF_sort = _ggDF.sort_values('count', ascending=False)

        return _ggDF_sort
    
    def scan(self, orig_seq='這邊的空氣污染很嚴重的市佔率'):
        # Consider the length of new lable
        
        seqs = [orig_seq[idx-(self.label_len-1):idx+1] for idx in range(self.label_len-1, len(orig_seq))]         if len(orig_seq) >= self.label_len else []
        
        self._check = dict((''.join(element[:-1]), element[-1]) for element in self.ptable.index.tolist())

        new = list(orig_seq)
        output = []
        for idx,s in enumerate(seqs, 1):
            flag = self._check.get(s)
            if flag:
                output.append((idx,flag))
                new[idx] = flag
        new = ''.join(new)

        return (new, output)
        


# In[17]:

def viterbi_ng(seq, par, show=0):
    # LM,NCM required
    para = par.get('lmNcm_weight', None)
    ng_num = par.get('ngnum', None)
    
    case = CASE(seq, ncm)
    
#     sect_named = namedtuple('sect_stat', 'score, length, idx, seq')
    
    cands4all =[]
    pre_cand_lst = []
    
    for cur_idx, cur_ch in enumerate(case.query):  
        section = []
        if show==1:
            print('==========')
            print(cur_idx, cur_ch, ' '.join([x[0] for x in case.cands[cur_idx]]))        

        if cur_idx==0 and cur_ch=='<s>':
            section.append({'from_to':'<s>', 'score':0.0, 'seq':['<s>'], 'idx_seq':[0]})

        elif cur_idx>0:                
            # cand:[0] for cand_ch, [1] for cand_prob
            for cand_idx, cand in enumerate(case.cands[cur_idx]):
                # Add global 
                sect_ncm = math.log10(cand[1])              
            
                batch = []
                
                # lm.scoring(list)
                for pre_idx, pre_cand in enumerate(pre_cand_lst):
                    batch_seq = pre_cand['seq'] + [cand[0]]
                    
                    # Compute before?
                    batch_lm = lm.scoring(batch_seq[-ng_num:], ng_num) + pre_cand['score']
                    batch_score = (para[0] * batch_lm + para[1] * sect_ncm)
                    
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
                wp.write(', 0')
                
            wp.write('\n')
                


# In[6]:

def debug_ncm(ch, append=None, value= 0.05, show=0):    
    tt = ncm.get_cands(ch)
    if show==1:
        for d in tt:
            print(d)
        
    if append:
        ncm.table[ch][append] = value
        if show==1:
            print('== Add %s to Set of %s' %(append,ch))
    else:
        return tt


# In[7]:

def debug_lm(seq, ngnum=2):
    lst = seq.split()
    for item in lst:
        print(item, lm.scoring(item, ngnum))


# In[8]:

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


# In[9]:

def batch(seq, par = {
        'lmNcm_weight':[0.7,0.3],
        'ngnum':3,
        'pre_con':True,
        'con_log_file':'special_case4con.txt',
        'show':0        
    }, show=0):            
    
    con_log_file = par.get('con_log_file', 'special_case4con.txt')
    
    sub_seqs = seperateSeq(seq)
    
    total_length = 0
    error_dict = dict()
    for sub in sub_seqs:
        if show==1: 
            print('pre:', sub, len(sub))
        '''
        Preprocess
        '''
        if par.get('pre_con',False):
            (sub, output) = con_preprocess.scan(sub)
            if len(output)>0: 
                with open(con_log_file,'a',encoding='utf8') as wp:
                    wp.write(sub)
                    for (idx,ch) in output:
                        wp.write('{}\t{}\t'.format(str(idx),ch))
                    wp.write('\n')
            con_dict = dict((str(idx+total_length+1), ch) for idx,ch in output)
            error_dict.update(con_dict)
        
        '''
        Viterbi
        '''
        if show==1: print(sub)
        result = viterbi_ng(sub, par, show)
        tmp_pos   = [i for i,e in enumerate(result['cor_idx']) if e!=0]
        pos_error = [str(idx+total_length+1) for idx in tmp_pos]     
        ch_error  = [result['correct'][int(idx)] for idx in tmp_pos]
        
        if show==1:
            for idx in tmp_pos:
                print(result['correct'][int(idx)])
            print(result['orig'])
            print(result['correct'])  
        
        viter_dict = dict((p,s) for p,s in zip(pos_error, ch_error))
        error_dict.update(viter_dict)
        
        '''
        End
        '''
        total_length += len(sub)
        
    return sorted(error_dict.items(), key=lambda x:int(x[0]))


# debug_lm('市占率 視障率')
# 
# lm.scoring('市占率',show=1)
# 
# lm.scoring('視障率',show=1)

# In[11]:

lm = LM(lm_filename)
ncm = NCM(channel_filename)
con_preprocess = CONFUSION(con_filename, con_log_file='special_case4con.txt')
con_preprocess.organize(label=['pre','error','corr'], threshold=10)


# In[ ]:

def t1(sys, par):
    if len(sys.argv) < 3:
        print('Usage: python filename.py token test_file1')
        sys.exit(1)
    else:
        token     = sys.argv[1]
        test_data = sys.argv[2]
        
        ncm_insert_vals = [0.005,0.01,0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85]

        for ncm_insert_val in ncm_insert_vals:
            result_name = './test_15/re_{}_{}.txt'.format
            ncm_tag = str(ncm_insert_val)[2:]
            
#             del ncm
            channel_filename = './confusionAdd/confusionSet_{}.pkl'.format(ncm_tag)
            ncm = NCM(channel_filename)

            run_test(test_data, result_name(token, ncm_tag), par)
                     


# In[ ]:

def t2(sys, par):
    global ncm
    if len(sys.argv) < 3:
        print('Usage: python filename.py token channel_model test_data')
        sys.exit(1)
    else:
        token     = sys.argv[1]
        channel_model = sys.argv[2]
        test_data = sys.argv[3]
        
        del ncm
        
        ncm = NCM(channel_model)
        
        result_name = './test_15/re_{}.txt'.format
               
        run_test(test_data, result_name(token), par)



# In[ ]:

def t3(sys, par):
    global lm
    if len(sys.argv) < 4:
        print('Usage: python filename.py token language_model test_data ngram')
        sys.exit(1)
    else:
        token     = sys.argv[1]
        language_model = sys.argv[2]
        test_data = sys.argv[3]
        par['ngnum'] = int(sys.argv[4])
        
        del lm 
        
        lm = LM(language_model)
        
        result_name = './test_15/re_{}.txt'.format
        run_test(test_data, result_name(token), par)


# In[ ]:

def t4(sys, par):
    global ncm
    if len(sys.argv) < 4:
        print('Usage: python filename.py token ncm_global channel_model test_data ')
        sys.exit(1)
    else:
        token = sys.argv[1]
        ncm_global = sys.argv[2]
        channel_model = sys.argv[3]
        test_data = sys.argv[4]
        
        del ncm
        
        ncm = NCM(channel_model, ncm_global)
        
        result_name = './test_15/re_{}.txt'.format
               
        run_test(test_data, result_name(token), par)


# In[13]:

if __name__=='__main__':
    par = {
        'lmNcm_weight':[0.7,0.3],
        'ngnum':3,
        'pre_con':False,
        'con_log_file':'special_case4con.txt',        
        'show':0
    }
    
    t4(sys,par)
    
    

