
# coding: utf-8

# In[1]:

import charSpellingCheck as CSC
import csv


# In[2]:

from tqdm import trange


# In[3]:

testData_name = './test_15/SIGHAN15_CSC_TestInput.txt'
testGroundTruth_name = './test_15/SIGHAN15_CSC_TestTruth.txt'
systemTruth_name = './test_15/re_newLM15_2.txt'


# In[7]:

output_name ='./confusionAdd/test_{}.csv'.format
log_file = 'log.txt'


# In[5]:

def toDict(lst):
    out = dict()
    if len(lst)%2==0:
        for idx in range(0, len(lst), 2):
            out[lst[idx]] = lst[idx+1]
    return out


# In[11]:

def task(gt_dict,st_dict,seq):
    # Original character
    for error_idx, corr_ch in gt_dict.items():
        error_ch = seq[int(error_idx)-1]
        sys_ch = st_dict.get(error_idx,'x')

        if corr_ch != sys_ch or error_idx not in st_dict:
            check = [i for i,_ in CSC.debug_ncm(error_ch) if i==corr_ch]
            if len(check)!=0: # 
                continue 

            total += 1
            if CSC.ncm.table.get(error_ch,None):
                CSC.debug_ncm(error_ch, corr_ch, ncm_insert_val)
            else:
                CSC.ncm.table[error_ch] = {corr_ch:ncm_insert_val, error_ch:0.95}

            new = CSC.batch(seq)
            new_st = dict((idx,err) for idx, err in new)

            row_write = [dataID,0,str(error_idx),error_ch,corr_ch,sys_ch,seq]
            if new_st.get(error_idx, None) == corr_ch:
                good += 1
                row_write[1] = 1
            CSC.ncm.table[error_ch].pop(corr_ch, None)
    return row_write


# In[13]:


ncm_insert_vals = [0.005,0.01,0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]
for ncm_insert_val in ncm_insert_vals:
    with open(testData_name, 'r', encoding='utf8') as tdp,    open(testGroundTruth_name, 'r', encoding='utf8') as tgtp,    open(systemTruth_name, 'r', encoding='utf8') as stp,    open(output_name(str(ncm_insert_val)[2:]), 'w', newline='',encoding='utf8') as ws:

    # open(output_name, 'w', encoding='utf8o') as wp:

        wp = csv.writer(ws)
        wp.writerow(['DataID','Label','Position','Error','Groud Truth','Original System Result','Sequence'])
    #     row_write = [dataID,0,str(error_idx),error_ch,corr_ch,sys_ch,seq]


        good, total = 0, 0
        for idx in trange(1100, unit='seq'):
            
            if idx>5: break
            
            td_line = tdp.readline().strip('\n')

            dataID = td_line[:(td_line.find(')'))+1]
            seq = td_line[(td_line.find(')')+2):]

#             tgt_line = tgtp.readline()
#             st_line  = stp.readline()
            gt = tgtp.readline().strip().split(', ')[1:]
            st = stp.readline().strip().split(', ')[1:]


            if gt == st:
                continue

            gt_dict = toDict(gt)
            st_dict = toDict(st)

            # Original character
            for error_idx, corr_ch in gt_dict.items():
                
                '''
                error_idx, error_ch
                corr_ch, sys_ch
                
                st_dict
                
                ncm_insert_val
                
                dataID, seq
                good, total
                '''
                
                error_ch = seq[int(error_idx)-1]
                sys_ch = st_dict.get(error_idx,'x')
                
                if corr_ch != sys_ch or error_idx not in st_dict:
                    check = [i for i,_ in CSC.debug_ncm(error_ch) if i==corr_ch]
                    if len(check)!=0: # 
                        continue 

                    total += 1
                    if CSC.ncm.table.get(error_ch,None):
                        CSC.debug_ncm(error_ch, corr_ch, ncm_insert_val)
                    else:
                        CSC.ncm.table[error_ch] = {corr_ch:ncm_insert_val, error_ch:0.95}

                    new = CSC.batch(seq)
                    new_st = dict((idx,err) for idx, err in new)

                    row_write = [dataID,0,str(error_idx),error_ch,corr_ch,sys_ch,seq]
                    if new_st.get(error_idx, None) == corr_ch:
                        good += 1
                        row_write[1] = 1
                    CSC.ncm.table[error_ch].pop(corr_ch, None)

                    wp.writerow(row_write)


    print('Done.')
    with open(log_file, 'a', encoding='utf8') as fp:
        fp.write('{}\t{}\t{}'.format(ncm_insert_val,good,total))
    print(good, total)


# import pandas as pd
# 
# tmp = pd.DataFrame.from_csv('confusionAdd_utf8.csv')
# special = tmp[tmp['Label']==0]
# special_case = special.loc[:,'Sequence']
# 
# tt = special.sample(1)
# tt
# 
# outputAll.to_csv('./zzz/output.csv')
# 
# outputAll = outputAll.append(tt)
# outputAll
