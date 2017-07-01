
# coding: utf-8

# In[1]:

import argparse
import requests
import os 
import pickle
import math
from collections import defaultdict
import re
import socket
import xml.etree.ElementTree as ET
from copy import deepcopy

import pandas as pd 

from model.model import LM, CASE, NCM, CKIP

import time 
import sys, getopt


# In[2]:

# data_root = '/home/kiwi/udn_data/training_confusion/'
# data_root = 'G:/UDN/training_confusion/'

if os.environ.get('USERDOMAIN') == 'KIWI-PC':
    data_root = 'G:/UDN/training_confusion/'
else:
    data_root = '/home/kiwi/udn_data/training_confusion/'

channel_filename = data_root+'channelModel.pkl'
lm_filename = data_root+'sinica.corpus.seg.char.lm'
con_filename = './extractUDN_prepost/all.csv'

# candsExpand = CANDSEXPAND(ncmEx_filename)


# In[4]:

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
        


# In[138]:




# In[53]:

def reRanking(candidates, origSeq, par):
    WEIGHT = par.get('WEIGHT', 0.7)
    NGNUM = par.get('NGNUM', 3)
    SEGMETHOD, LMMETHOD = par.get('RERANK', ['CKIP', 'ALL'])
    CKIPACCOUNT = par.get('CKIP',{'username':'sean2249','password':'3345678'})
    WORDPORT = par.get('WORDPORT', 5488)
    SHOW = par.get('SHOW', 1)

    ckip = CKIP(**CKIPACCOUNT)

    lm_get = 'http://140.112.91.62:{}/api/{}{}'.format
    re_candidates = []
    for (cand, chScore,_) in candidates:        
        candSeq = ''.join(cand[1:-1])

        if SEGMETHOD == 'JIEBA':
            candSeg = list(jieba.cut(candSeq))
        elif SEGMETHOD == 'CKIP':    
            candSeg = list(ckip.cut(candSeq))

        ncmScore = 0.0
        for o, c in zip(origSeq, candSeq):
            if o != c:
                try:
                    ncmScore += math.log10(ncm.table[o][c])
                except:
                    print(o,c)
                    print(origSeq, candSeq)
                    raise RuntimeError('ass')

        # Language model
        if LMMETHOD == 'ELEMENT':
            wordLM = 0
            for item in candSeg:
                i = [item]
                wordLM += requests.get(lm_get(WORDPORT, NGNUM, '||'.join(i))).json()['score']
        elif LMMETHOD == 'ALL':
            wordLM = requests.get(lm_get(WORDPORT, NGNUM, '||'.join(candSeg))).json()['score']    

        wordScore = wordLM*WEIGHT + ncmScore*(1-WEIGHT)

        re_candidates.append((candSeg, chScore, wordScore))

    
    if SHOW == 1:        
        wordCands = sorted(re_candidates, key= lambda x:x[2], reverse=True)
        chCands = sorted(re_candidates, key= lambda x:x[1], reverse=True)
        for w,c in zip(wordCands, chCands):
            print('{}/{:.2f}\t{}/{:.2f}'.format(' '.join(w[0]), w[2], ''.join(c[0]), c[1]))
        
    return sorted(re_candidates, key= lambda x:x[1]+x[2], reverse=True)[:3]
     


# In[6]:

def beamSearch(seq, par):
    
    SHOW = par.get('SHOW', 1)
    NGNUM = par.get('NGNUM', 3)
    WEIGHT = par.get('WEIGHT', 0.7)
    PRUNE_LIMIT = par.get('BATCH',[0, 20])[1]
    CHARPORT = par.get('CHARPORT', 5487)     
    
    lm_get = 'http://140.112.91.62:{}/api/{}{}'.format
    case = CASE(seq, ncm)    
    
    case.query[-1] = '<s>'
    stack = [(case.query, requests.get(lm_get(CHARPORT, NGNUM, '||'.join(case.query))).json()['score'], 0)]
    
    for cur_idx, cur_ch in enumerate(case.query):
        batch = []
        
        if SHOW==1:
            print('========')            
            print(cur_idx, ' '.join([x[0] for x in case.cands[cur_idx]]))
             
        for (cand, cand_prob) in case.cands[cur_idx]:                                        
            cand_ncm = math.log10(cand_prob)

            for cur_seq, orig_score, orig_ncm in stack:                   
                sect_seq = list(cur_seq)
                sect_seq[cur_idx] = cand
                
                if sect_seq[-1] == '</s>':
                        sect_seq[-1] = '<s>'
                try:
                    sect_lm = requests.get(lm_get(CHARPORT, NGNUM, '||'.join(sect_seq))).json()['score']
                except:
                    print('ERROR')
                    return 
                
                sect_ncm = orig_ncm + cand_ncm

                sect_score = (sect_lm*WEIGHT) + (sect_ncm*(1-WEIGHT))
                
                batch.append((sect_seq, sect_score, sect_ncm))

        stack = sorted(batch, key= lambda x:x[1], reverse=True)[:PRUNE_LIMIT]
            
    return stack


# In[7]:

def viterbi(seq, par):
    SHOW = par.get('SHOW', 1)
    NGNUM = par.get('NGNUM', 3)
    WEIGHT = par.get('WEIGHT', 0.7)
    LMMETHOD = par.get('BATCH',[0, 'ELEMENT'])[1]
    CHARPORT = par.get('CHARPORT', 5487)        
    
    # lm http    
    lm_get = 'http://140.112.91.62:{}/api/{}{}'.format        
    
    case = CASE(seq, ncm)
    
    for cur_idx, cur_ch in enumerate(case.query):  
        if SHOW==1:
            print('==========')
            print(cur_idx, ' '.join([x[0] for x in case.cands[cur_idx]]))        

        if cur_idx==0:
            section = [(['<s>'], 0.0, 0)]
                            
        else:
            tmp_section = []
            for (cand, cand_prob) in case.cands[cur_idx]:
                cand_ncm = math.log10(cand_prob)
                batch = []
                for (pre_seq, pre_score, pre_ncm) in section:
                    batch_seq = list(pre_seq)
                    batch_seq.append(cand)
                    
                    if batch_seq[-1] == '</s>':
                        batch_seq[-1] = '<s>'
                        
                    lmSeq = batch_seq if LMMETHOD == 'ALL' else batch_seq[-NGNUM:]
                    
                    try:
                        batch_lm = (requests.get(
                            lm_get(CHARPORT, NGNUM, '||'.join(lmSeq))).json()['score'])
                    except:
                        print(batch_seq)
                        print(lm_get(CHARPORT, NGNUM, '||'.join(batch_seq)))
                        return 
                    
                    batch_ncm = pre_ncm + cand_ncm
                    batch_score = batch_lm * WEIGHT + cand_ncm * (1-WEIGHT) + pre_score
                    
                    batch.append((batch_seq, batch_score, batch_ncm))
                    
                winner = max(batch, key=lambda x:x[1])
                if SHOW ==1:
                    print('{}  {}'.format(cand, winner[0]))
                tmp_section.append(winner)
                
            section = list(tmp_section)     
    
    
    sub = section[0][0]
    return ''.join(sub[1:-1])
                    


# In[8]:

def run_test(testfilename, resultname, par):    
    show = par.get('SHOW',0)
    
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
                


# In[9]:

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


# In[10]:

def debug_lm(seq, ngnum=2):
    lst = seq.split()
    for item in lst:
        print(item, lm.scoring(item, ngnum))


# In[11]:

def seperateSeq(seq):
    pattern = re.compile('[，。！；]')
    
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


# In[12]:

def batch(seq, par):
    SHOW = par.get('SHOW', 1)
    PRE  = par.get('PRE', False)
    METHOD = par.get('BATCH', ['BEAM','ALL'])[0]
    
    
    con_log_file = par.get('con_log_file', 'special_case4con.txt')
    
    sub_seqs = seperateSeq(seq)
    
    total_length = 0
    error_dict = dict()
    for orig in sub_seqs:
        if SHOW==1: 
            print('Original: {}'.format(orig))
        '''
        Preprocess
        '''
        sub = str(orig)
        if PRE:
            (sub, errors) = con_preprocess.scan(orig)
            erros = dict((str(idx+total_length+1), ch) for idx, ch in errors)
            errro_dict.update(errors)
            if SHOW == 1:
                print('Pre: {}'.format(sub))                
            
        if METHOD == 'VITERBI':
            
            sub = viterbi(sub, par)
            if SHOW == 1:
                print('Viterbi: {}'.format(sub))
                
        elif METHOD == 'BEAM':
            candidates = beamSearch(sub, par)
            sub = reRanking(candidates, orig, par)[0][0]
            sub = ''.join(sub)
            if SHOW == 1:
                print('Beam: {}'.format(sub))
            
        errors = dict((str(idx+total_length+1), s) 
              for idx, (o, s) in enumerate(zip(orig, sub)) if o != s)            
        error_dict.update(errors)                                    
        total_length += len(sub)
        
        print(sub)
        
    return sorted(error_dict.items(), key=lambda x:int(x[0]))

# In[14]:

word_filename = data_root + 'dict_word.txt'
# ref_word = CASE()


# In[15]:

# lm = LM(lm_filename)
# ncm = NCM(channel_filename)
ncm = NCM('./confusionTable/confu_999995_9_50.pkl')
# con_preprocess = CONFUSION(con_filename, con_log_file='special_case4con.txt')
# con_preprocess.organize(label=['pre','error','corr'], threshold=10)


# In[16]:

def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    parser.add_argument('--test', required=True)
    
    parser.add_argument('--lm', default=0)
    parser.add_argument('--ngnum', type=int, default=2)
    
    parser.add_argument('--ncm', default=0)
    
    return parser.parse_args()


# In[17]:

def main(args, par):
    result_name = './test_15/re_{}.txt'.format
    
    if args.lm != 0:
        del lm 
        lm = LM(args.lm)
    if args.ncm != 0:
        del ncm
        ncm = NCM(args.ncm)
    
    
    par['ngnum'] = args.ngnum
    
    
    run_test(args.test, result_name(args.token), par)


# In[18]:

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
                     


# In[19]:

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



# In[20]:

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


# In[21]:

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


# In[74]:

def udn_t1():
    with open('UDN_benchmark/test.txt', 'r', encoding='utf8') as fp,\
        open('dd.log', 'w', encoding='utf8') as wp:
        wp.write('Seq|||GroundTruth|||System\n')
        line = fp.readlines()[11]        
        seq, error_info = line.strip().split('|||')

        sys_info = batch(seq, par)

        print(seq, error_info, sys_info)

        if sys_info != error_info:
            wp.write('x|||{}|||{}|||{}\n'.format(seq, error_info, sys_info))
        else:
            wp.write('o|||{}|||{}|||{}\n'.format(seq, error_info, sys_info))



# In[76]:

if __name__=='__main__':
    #     args = process_command()
    par = {
        'WORDPORT':5488,
        'CHARPORT':5487,        
        'PRE':False,
        'NGNUM':3,
        'WEIGHT':0.7,
        # [viterbi, all/element]
        # [beam, prune_limit]
#         'BATCH':['VITERBI', 'ALL'], 
        'BATCH':['BEAM', 10],
        # CKIP/JIEBA ALL/ELEMENT
        'RERANK':[
            'CKIP','ALL'],
        'CKIP':{'username':'sean2249', 'password':'3345678'},
        'SHOW':0              
    }
    
    udn_t1()


# In[202]:

def tmp():
    # seq = '幸虧我會說德問'
    # seq = '因為他很用功常常坐最'
    # seq = '花村的地途'
    seq = '能利用積木作出機器人的結構，'
    seqs = [
        '已經有別於已往，', #(以)
        '聯合國世界銀行國際金融公司與英國金融時報於台北時間今天清晨宣布新北市獲得「城市轉型卓越奬」，', #(獎)
        '未能激起配偶性慾是女性最常犯的大錯之一。', #(欲)
        '「林右昌在基層很紮實、實在，', #(扎)
        '不要讓學生覺得沒唸書也會及格，', #(念)
        '目前委託台灣大學資工系副教授廖世偉協助瞭解是否有機會可以做成。', #(了)
        '區塊鏈系統也兩大特色，', #(有)
        '青創先峰匯旨在為兩岸青年搭建平台切磋，', #(鋒)
        '規劃保險時，', #(畫)
    ]

    for idx, seq in enumerate(seqs):
        print(idx, seq)
        par = {'BATCH':['VITERBI','ALL'], 'SHOW':0}
        batch(seq, par)
        par = {'BATCH':['VITERBI','ELEMENT'], 'SHOW':0}
        batch(seq, par)
        par = {'BATCH':['BEAM', 20], 'RERANK':['CKIP','ALL'], 'SHOW':0}
        batch(seq, par)
        par = {'BATCH':['BEAM', 20], 'RERANK':['CKIP','ELEMENT'], 'SHOW':0}
        batch(seq, par)



