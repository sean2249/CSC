
# coding: utf-8

# In[71]:

import argparse
import os
import pickle
import math
# from collections import defaultdict
import re
import sys
# import socket
# import xml.etree.ElementTree as ET
# from copy import deepcopy
import requests

import pandas as pd
# import jieba

from model.model import CASE, NCM, CKIP


# In[72]:

if os.environ.get('USERDOMAIN') == 'KIWI-PC':
    DATA_ROOT = 'G:/UDN/training_confusion/'
else:
    DATA_ROOT = '/home/kiwi/udn_data/training_confusion/'


# In[73]:

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


# In[74]:

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
        


# In[204]:

a = s[:10]


# In[205]:

seq


# In[212]:

batch(seq, par)


# In[209]:

reRanking(a, seq, par)


# In[211]:

def reRanking(candidates, origSeq, par):
    WEIGHT = par.get('WEIGHT', 0.7)
    NGNUM = par.get('NGNUM', 3)
    SEGMETHOD, LMMETHOD = par.get('RERANK', ['CKIP', 'ALL'])
    CKIPACCOUNT = par.get('CKIP',{'username':'sean2249','password':'3345678'})
    WORDPORT = par.get('WORDPORT', 5460)
    SHOW = par.get('SHOW', 1)
    LMIP = par.get('LMIP', '140.114.77.143')

    if SEGMETHOD =='CKIP':
        ckip = CKIP(**CKIPACCOUNT)

    lm_get = 'http://{}:{}/api/{}{}'.format
    re_candidates = []
    for (cand, chScore,_) in candidates:
        candSeq = str(cand)

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
                wordLM += requests.get(lm_get(LMIP, WORDPORT, NGNUM, '||'.join(i))).json()['score']
        elif LMMETHOD == 'ALL':
            wordLM = requests.get(lm_get(LMIP, WORDPORT, NGNUM, '||'.join(candSeg))).json()['score']    

        wordScore = wordLM*WEIGHT + ncmScore*(1-WEIGHT)

        re_candidates.append((candSeg, chScore, wordScore))

    
    if SHOW == 1:        
        wordCands = sorted(re_candidates, key= lambda x:x[2], reverse=True)
        chCands = sorted(re_candidates, key= lambda x:x[1], reverse=True)
        for w,c in zip(wordCands, chCands):
            print('{}/{:.2f}\t{}/{:.2f}'.format(' '.join(w[0]), w[2], ''.join(c[0]), c[1]))
    
    best = max(re_candidates, key=lambda x:x[1]+x[2])
    
    return ''.join(best[0])
        
#     return sorted(re_candidates, key= lambda x:x[1]+x[2], reverse=True)[:5]
     


# In[221]:

def beamSearch(seq, par):
    
    SHOW = par.get('SHOW', 1)
    NGNUM = par.get('NGNUM', 3)
    WEIGHT = par.get('WEIGHT', 0.7)
    PRUNE_LIMIT = par.get('METHOD',[0, 20])[1]
    CHARPORT = par.get('CHARPORT', 5480)     
    LMIP = par.get('LMIP', '140.114.77.143')

    
    lm_get = 'http://{}:{}/api/{}{}'.format
    case = CASE(seq, ncm)    
    
    case.query[-1] = '<s>'
    stack = [(case.query, requests.get(lm_get(LMIP, CHARPORT, NGNUM, '||'.join(case.query))).json()['score'], 0)]
    
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
                    sect_lm = requests.get(lm_get(LMIP, CHARPORT, NGNUM, '||'.join(sect_seq))).json()['score']
                except:
                    print('ERROR')
                    return 
                
                sect_ncm = orig_ncm + cand_ncm
                
                sect_score = (sect_lm*WEIGHT) + (sect_ncm*(1-WEIGHT))
                
#                 print(sect_seq, sect_lm, sect_ncm, sect_score)
                
                batch.append((sect_seq, sect_score, sect_ncm))

        stack = sorted(batch, key= lambda x:x[1], reverse=True)[:PRUNE_LIMIT]       
           
    return [(''.join(s[1:-1]), score, ncm_score) 
            for s, score, ncm_score in stack]


# In[77]:

def viterbi(seq, par):
    SHOW = par.get('SHOW', 1)
    NGNUM = par.get('NGNUM', 3)
    WEIGHT = par.get('WEIGHT', 0.9)
    LMMETHOD = par.get('METHOD',[0, 'ELEMENT'])[1]
    CHARPORT = par.get('CHARPORT', 5480)    
    LMIP = par.get('LMIP', '140.114.77.143')
    
    # lm http    
    lm_get = 'http://{}:{}/api/{}{}'.format        
    
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
                            lm_get(LMIP, CHARPORT, NGNUM, '||'.join(lmSeq))).json()['score'])
                    except:
                        print(batch_seq)
                        print(lm_get(LMIP, CHARPORT, NGNUM, '||'.join(batch_seq)))
                        return 
                    
                    batch_ncm = pre_ncm + cand_ncm
                    batch_score = batch_lm * WEIGHT + cand_ncm * (1-WEIGHT) + pre_score
                    
#                     print(lmSeq, batch_ncm, batch_lm, batch_score)
                    
                    batch.append((batch_seq, batch_score, batch_ncm))
                    
                winner = max(batch, key=lambda x:x[1])
                if SHOW ==1:
                    print('{}  {}'.format(cand, winner[0]))
                tmp_section.append(winner)
                
            section = list(tmp_section)     
    
    
    sub = section[0][0]
    return ''.join(sub[1:-1])


# In[179]:

def viterbiN(seq, par):
    SHOW = par.get('SHOW', 1)
    NGNUM = par.get('NGNUM', 3)
    WEIGHT = par.get('WEIGHT', 0.9)
    LMMETHOD = par.get('METHOD',[0, 'ELEMENT'])[1]
    CHARPORT = par.get('CHARPORT', 5480)    
    LMIP = par.get('LMIP', '140.114.77.143')
    
    LIMIT = par.get('VITERLIMIT', 2)
    
    # lm http    
    lm_get = 'http://{}:{}/api/{}{}'.format        
    
    case = CASE(seq, ncm)
    
    for cur_idx, cur_ch in enumerate(case.query[:-1]):  
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
                            lm_get(LMIP, CHARPORT, NGNUM, '||'.join(lmSeq))).json()['score'])
                    except:
                        print(batch_seq)
                        print(lm_get(LMIP, CHARPORT, NGNUM, '||'.join(batch_seq)))
                        return 
                    
                    batch_ncm = pre_ncm + cand_ncm
                    batch_score = batch_lm * WEIGHT + cand_ncm * (1-WEIGHT) + pre_score
#                     print(lmSeq, batch_ncm, batch_lm, batch_score)
                    
                    batch.append((batch_seq, batch_score, batch_ncm))
                    
                nbest = sorted(batch, key=lambda x:x[1], reverse=True)[:LIMIT]
                tmp_section.extend(nbest)

                
            section = list(tmp_section)   
    
    return [(''.join(seq[1:]), score, ncm_score)
            for seq, score, ncm_score in sorted(section, key=lambda x:x[1], reverse=True)]


# In[95]:

# secretRule
def secretRule(seq):
    rule_dict = {
#         '週':'周',
#         '臺':'台'
    }
    sub = str(seq)
    for error, correct in rule_dict.items():
        if error in sub:
            sub = sub.replace(error, correct)
    return sub


# In[78]:

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


# In[79]:

def debug_lm(seq, ngnum=2):
    lst = seq.split()
    for item in lst:
        print(item, lm.scoring(item, ngnum))


# In[224]:

def batch(seq, par):
    SHOW = par.get('SHOW', 1)
    PRE  = par.get('PRE', False)
    RULE = par.get('RULE', False)
    METHOD = par.get('METHOD', ['BEAM','ALL'])[0]
    RERANK = par.get('RERANK', [])
    
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
            con_log_file = par.get('con_log_file', 'special_case4con.txt')
            (sub, errors) = con_preprocess.scan(orig)
            erros = dict((str(idx+total_length+1), ch) for idx, ch in errors)
            errro_dict.update(errors)
            if SHOW == 1:
                print('Pre: {}'.format(sub))
        if RULE:            
            sub = secretRule(sub)
            orig = str(sub)
            
        if METHOD == 'VITERBI':
            if not RERANK:
                sub = viterbi(sub, par)
            else:
                subs = viterbiN(sub, par)
                sub = reRanking(subs, orig, par)
            if SHOW == 1:
                print('Viterbi: {}'.format(sub))
        elif METHOD == 'BEAM':
            candidates = beamSearch(sub, par)
            if not RERANK:
                sub = ''.join(candidates[0][0])
            else:            
                sub = reRanking(candidates, orig, par)                
            if SHOW == 1:
                print('Beam: {}'.format(sub))
            
        errors = dict((str(idx+total_length+1), s) 
              for idx, (o, s) in enumerate(zip(orig, sub)) if o != s)            
        error_dict.update(errors)                                    
        total_length += len(sub)
        
    return sorted(error_dict.items(), key=lambda x:int(x[0]))


# In[81]:

def runSIGHANtest(testfilename, resultname, par):    
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
                


# In[18]:

def runUDNtest(testfilename, resultname, par):
    with open(testfilename, 'r', encoding='utf8') as fp,    open(resultname, 'w', encoding='utf8') as wp:
        wp.write('Seq|||GroundTruth|||System\n')
        for idx, line in enumerate(fp):            
            seq, error_info = line.strip().split('|||')
            print(idx, seq)            
            sys_info = batch(seq, par)
            sys_info = ', '.join(['{}, {}'.format(idx, ch) for idx, ch in sys_info])                        

            if sys_info != error_info:
                wp.write('x|||{}|||{}|||{}\n'.format(seq, error_info, sys_info))
            else:
                wp.write('o|||{}|||{}|||{}\n'.format(seq, error_info, sys_info))


# In[15]:

ncm = NCM('./confusionTable/0601/confu_999995_9_50.pkl')
# con_preprocess = CONFUSION(con_filename, con_log_file='special_case4con.txt')
# con_preprocess.organize(label=['pre','error','corr'], threshold=10)


# In[122]:

def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    parser.add_argument('--test', type=str, default='0')
    
    parser.add_argument('--rule', action='store_true')
    parser.add_argument('--method', type=str, required=True, help='v(iterbi) OR b(eam)')
    parser.add_argument('--methodsmall', type=str, required=True, help='v[element OR all] b[limit_number]')
    parser.add_argument('--rerank', type=str, default='0', help='jieba or ckip')
    
    parser.add_argument('--ngnum', type=int, default=2)
    parser.add_argument('--weight', type=float, default=0.7)
    parser.add_argument('--chport', type=int, default=5480)
    parser.add_argument('--wordport', type=int, default=5460)
    
    parser.add_argument('--ncm', default=0)
    
    return parser.parse_args()


# In[17]:

def main(args, par):
    if args.ncm != 0:
        del ncm
        ncm = NCM(args.ncm)
    
    par['NGNUM'] = args.ngnum
    par['WEIGHT'] = args.weight    
    par['CHARPORT'] = args.chport
    par['WORDPORT'] = args.wordport
    par['RULE'] = args.rule
    par['RERANK'] = args.rerank
    testfile = args.test
    
    if args.method == 'v':
        if args.methodsmall == 'element':
            par['METHOD'] = ['VITERBI','ELEMENT']
        elif args.methodsmall == 'all':
            par['METHOD'] = ['VITERBI', 'ALL']    
        else:
            print('Wrong parameter')
            sys.exit(0)
    elif args.method == 'b':
        if args.methodsmall.isdigit():
            par['METHOD'] = ['BEAM', int(args.methodsmall)]
        else:
            print('Wrong parameter')
            sys.exit(0)
    else:
        print('Wrong parameter')
        sys.exit(0)        
        
    if args.rerank == '0':
        par['RERANK'] = []
    elif args.rerank.lower() == 'jieba':
        par['RERANK'] = ['JIEBA', 'ALL']
    elif args.rerank.lower() == 'ckip':
        par['RERANK'] = ['CKIP', 'ALL']
    else:
        print('Wrong parameter')
        sys.exit(0)
    
    if args.test == '0' or 'UDN' in args.test:
        if args.test == '0':
            testfile = './UDN_benchmark/testdata/UDN_testdata.txt'
        result_name = './UDN_benchmark/re_{}.txt'.format
        runUDNtest(testfile, result_name(args.token), par)
    else:
        result_name = './test_15/re_{}.txt'.format
        runSIGHANtest(args.test, result_name(args.token), par)


# In[ ]:

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




# In[19]:

if False:
    # Rule-based 
    rule_chfile = './confusionTable/rule_chPairs.pkl'
    rule_wordfile = './confusionTable/rule_wordPairs.pkl'

    pairs_ch = pickle.load(open(rule_chfile, 'rb'))
    pairs_word = pickle.load(open(rule_wordfile, 'rb'))

    # pairs_cntword = [dict()] * 10
    pairs_cntword = []
    idx = 0 
    for corr, errors in pairs_word.items():
    #     if idx > 3: 
    #         break
    #     else:
    #         idx += 1
        level = len(corr)           
    #     print(corr, errors, level)

        while (level>=len(pairs_cntword)):
            pairs_cntword.append(dict())


        for error in errors:
    #         print(pairs_cntword)
            pairs_cntword[level][error]=corr    


    pairs_cntword[2]

    seq = '我的褓母喜歡喝牛奶'

    for s in zip(seq[::1],seq[1::1]):
        print(s)

    pairs_cntword[2].get('褓母')

    for gram_cnt in range(len(pairs_cntword)):
        check_seqs = [seq[idx:idx+gram_cnt:] for idx in range(0,len(seq)-gram_cnt+1)]
    #     check_flag = [pairs_cntword[gram_cnt].get(s,'') for s in check_seqs ]
    #     print(check_flag)
        check_flag = []
        for idx, s in enumerate(check_seqs):
            t = pairs_cntword[gram_cnt].get(s)        

            if t:
                check_seqs[idx] = t
                check_flag.append((idx,t))

        print(check_seqs)
    #     print(check_flag)

    #     for idx, newword in check_flag


# In[20]:

if __name__=='__main__':
    par = {
        'LMIP':'140.114.77.143',
        'WORDPORT':5488,
        'CHARPORT':5487,        
        'PRE':False,
        'RULE':True,
        'NGNUM':3,
        'WEIGHT':0.7,
        'METHOD':['BEAM', 10],
        
        # CKIP/JIEBA ALL/ELEMENT
        'RERANK':[
            'CKIP','ALL'],
        'CKIP':{'username':'sean2249', 'password':'3345678'},
        'SHOW':0  
    }
    args = process_command()
    main(args, par)
    

