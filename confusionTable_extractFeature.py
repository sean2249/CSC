
# coding: utf-8

# ## Collect all confusion set

# In[69]:

import os 
import sys
import pickle
import pandas as pd
import csv
from collections import defaultdict
from bs4 import BeautifulSoup
import xlrd
import multiprocessing

import random
import time 


# ## Big unihan

# In[2]:

def bigUnihan_extract(filename):
    df = pd.read_csv(filename, sep='|', low_memory=False)
    df = df[['char','kFrequency']].set_index('char')
    df = df[~pd.isnull(df.kFrequency)]
    
    return df.to_dict()['kFrequency']


# ## SIGHAN_char_information

# In[3]:

def shape_compare_SIGHAN(ch_x, ch_y):
    '''
    return (similar, 同部首同筆畫數)
    '''
    cands1 = shape_SIGHAN.get(ch_x, [])
    try:
        cands2 = sound_SIGHAN.loc[ch_x].同部首同筆畫數
        if type(cands2)==float:
            cands2 = []
    except KeyError:
        cands2 = []
    
    out1 = 1 if ch_y in cands1 else 0
    out2 = 1 if ch_y in cands2 else 0
    
    return (out1, out2)


# In[4]:

def sound_compare_SIGHAN(ch_x, ch_y):
    '''
    4. 同音同調
    3. 同音異調
    2. 近音同調
    1. 近音異調
    0 Not Found  
    '''
    try:
        row = sound_SIGHAN.loc[ch_x]
        for idx, col in enumerate(row[:-1]):
            if type(col)==str and col.find(ch_y)!=-1:
                return 4-idx
                break
        else:
            return 0
    except KeyError:
        return 0


# ## unihan.csv (注音跟倉頡)

# In[5]:

def unihan_extract(unihan_filename):
    global dicBPMF, dicPhone, dicCangjie, dicCangjie2char
    with open(unihan_filename, 'r', encoding='utf8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            row = [cell for cell in row] # unicode
            char, bpmf, cangjie, components, jp, kr, name, pinyin_chs, pinyin_cht, char_strokes_count, radical, radical_name, radical_strokes_count = row
            for ph in bpmf.split(): # 發音
                dicBPMF[char] += [ph]
                dicPhone[ph] += [char]
            for cj in cangjie.split(): # 倉頡碼
                if u"難" in cj: continue
                for i in range(0, 3):
                    if i == len(cj): continue
                    # dicBPMF[char]['cangjie'] += [cj[:i]]
                    dicCangjie[char] += [cj[i:]]
                    # dicCangjie[cj[:i]] += [char]
                    dicCangjie2char[cj[i:]] += [char]


# ## Big Unihan.csv

# ## zwt.titles.txt (字典)

# In[6]:

def zwtTitle_train(lines):
    d = defaultdict(lambda: 0)
    for word in lines:
        d[word.strip()] += 1
    #d[word.strip().decode('utf-8')[:2]] += 1
    #print word.strip().decode('utf-8')[:2]
    return d


# ## radical.txt (部首)

# In[7]:

def radicalDic(lines):
    dicRadicalnum = defaultdict(list)
    dicRadical = defaultdict(list)
    for line in lines:
        for char in line[5:].strip().split('|'):
            dicRadical[char] += [line[:4]]
            dicRadicalnum[line[:4]] += [char]
    return dicRadicalnum, dicRadical


# In[8]:

def shape_similar(char):
    return list(set(ch for rnum in dicRadical[char] for ch in dicRadicalnum[rnum]))


# # Tone extraction

# In[9]:

def sound_extract_same(char):
    '''
    Same neutral and tone
    '''
    return list(set(ch for ph in dicBPMF[char] for ch in dicPhone[ph]))


# In[10]:

def sound_extract_tone(char):
    '''
    the char of different tone 
    '''
    output = set()
    tones = ['ˊ', 'ˇ', 'ˋ', '˙']
    for ph in dicBPMF[char]:
        if ph[-1] in tones:
            for t in tones:
                if t == ph[-1]: continue
                output = output.union(dicPhone[ph[:-1]+t])
        else:
            for t in tones:
                output = output.union(dicPhone[ph+t])
    return output
        


# In[11]:

def sound_extract_finalConsonant(char, toneKeep=True):
    '''
    單：ㄚㄛㄜㄝ
    複：ㄞㄟㄠㄡ
    鼻：ㄢㄣㄤㄥ
    捲舌：ㄦ
    '''
    output = set()
    tones = ['','ˊ', 'ˇ', 'ˋ', '˙']
    consonants = [
        ['ㄚ','ㄛ','ㄜ','ㄝ'],
        ['ㄞ','ㄟ','ㄠ','ㄡ'],
        ['ㄢ','ㄣ','ㄤ','ㄥ']
    ]
    
    for ph in dicBPMF[char]:
        # Tone delete
        if ph[-1] in tones:
            neutral, tone = ph[:-1], ph[-1]
        else:
            neutral, tone = ph,''
        
        # Add relevent consonant
        for cons in consonants:
            if neutral[-1] in cons:
                new_neutrals = set(neutral[:-1] + c for c in cons if c!=neutral[-1])
                for n in new_neutrals:
                    if toneKeep:
                        output = output.union(dicPhone[n+tone])
                    else:
                        for t in tones:
                            output = output.union(dicPhone[n+t])
                break
                               
    return output


# In[12]:

def sound_extract_similartConsonant(char, toneKeep=True):
    '''
    一次只針對一種，不會並用
    Initial
    ㄈㄏ
    ㄋㄌ
    ㄓㄗ
    ㄔㄘ
    Final:
    ㄢㄤ
    ㄜㄦ
    ㄣㄥ
    Intermediate:
    ㄧㄩ
    '''
    new_neutrals = set()
    output = set()
    tones = ['','ˊ', 'ˇ', 'ˋ', '˙']
    initial_pairs = [
        ['ㄈ','ㄏ'],
        ['ㄋ','ㄌ'],
        ['ㄓ','ㄗ'],
        ['ㄔ','ㄘ']
    ]
    final_pairs = [
        ['ㄢ','ㄤ'],
        ['ㄜ','ㄦ'],
        ['ㄣ','ㄥ']
    ]
    inter_pairs = [['ㄧ','ㄩ']]
    
    for ph in dicBPMF[char]:        
        # Tone delete
        if ph[-1] in tones:
            neutral, tone = ph[:-1], ph[-1]
        else:
            neutral, tone = ph, ''
            
        # Initial-consonant, just pick one 
        for cons in initial_pairs:
            if neutral[0] in cons:
                new_neutrals = new_neutrals.union(c + neutral[1:] + tone for c in cons if c!=neutral[0])
                break        
        
                    
        # Final-consonant       
        for cons in final_pairs:
#             print(neutral[-1], cons)
            if neutral[-1] in cons:
#                 print('i', cons)
#                 print(neutral[:-1])
#                 print(list(neutral[:-1] + c for c in cons if c!=neutral[-1]))
                new_neutrals = new_neutrals.union(neutral[:-1] + c + tone for c in cons if c!=neutral[-1])
                break
        
        # Inter_
        for cons in inter_pairs:
            for idx, tmp in enumerate(neutral):
                if tmp in cons:
                    new_neutrals = new_neutrals.union(neutral[:idx] + c + neutral[idx+1:] + tone for c in cons if c!=tmp)
                    break
    
    ######## fIX TOne pRoblEm
#     print(new_neutrals)
    # Get candidate based on new_neutrals
    for n in new_neutrals:
        if toneKeep:
            output = output.union(dicPhone[n])
        else:
            tmp = n[:-1] if n[-1] in tones else n            
            for t in tones:
                output = output.union(dicPhone[n+t])
#         print(n,' '.join(output))
        
#     print(len(output))                     
    return output if len(output)>0 else []


# # Cangjie Extraction (from UNIHAN)

# In[13]:

def cangjie_extract_same(char):
    cang = dicCangjie[char]
    if len(cang) > 0:
        output = set(dicCangjie2char[cang[0]])
        output.remove(char)
    else:
        output = set()
    
    return list(output)


# In[14]:

def cangjie_compare_unihan(ch_x,ch_y):
    '''
    Compare the cangjie between two character
    applied LCS to check whether the two chars have similar cangjie code 
    '''
    
    def lcs(xstr, ystr):
        """
        >>> lcs('thisisatest', 'testing123testing')
        'tsitest'
        """
        if not xstr or not ystr:
            return ""
        x, xs, y, ys = xstr[0], xstr[1:], ystr[0], ystr[1:]
        if x == y:
            return x + lcs(xs, ys)
        else:
            return max(lcs(xstr, ys), lcs(xs, ystr), key=len)
    
    cang_x = dicCangjie.get(ch_x,[])
    cang_y = dicCangjie.get(ch_y,[])
    
    if len(cang_x)==0 or len(cang_y)==0:
        return 0
    else:
        cang_x == cang_x[0]
        cang_y == cang_y[0]
    
    if cang_x == cang_y:
        return 2
    else:
        lcs_length = len(lcs(cang_x, cang_y))
        if len(cang_x) == 2:
            if (lcs_length == 1 and len(cang_y)==2)            or (lcs_length == 2 and len(cang_y)==3):
                return 1
        elif len(cang_x) == 3:
            if lcs_length == 2 and len(cang_y)<=4:
                return 1
        elif len(cang_x) == 4:
            if lcs_length == 3 and len(cang_y)>=3:
                return 1
        elif len(cang_y) == 5:
            if lcs_length == 4 and len(cang_y)==4:
                return 1
    
    return 0     


# ## Error_correct pair

# In[15]:

def extractPairs(filelist):
    for filename, path in filelist.items():
        print('== Filename: {}'.format(filename))
        '''
        QQQQQQ 有兩個以上的錯誤在一個詞裡面，但更正只有一項
        把上方例子放棄不取
        有 duplicate 
        '''
        # 1新編常用錯別字門診.txt OR 4教育部錯別字表.txt
        if filename.startswith('1') or filename.startswith('4'):
            df = pd.read_csv(path, sep='\t')
        # 2東東錯別字.txt OR 3常見錯別字一覽表.txt
        elif filename.startswith('2') or filename.startswith('3'):        
            df = pd.read_csv(path, sep='\t', header=None, names = ['正確詞','錯誤詞','正確字','錯誤字'])
        elif filename.startswith('udn_common'):
            table = xlrd.open_workbook(path).sheet_by_index(0)
            ch_dict = defaultdict(set)
            # Have multierros (error_word to correct_word)
            word_dict = defaultdict(set)
            for idx in range(1,table.nrows):
                row = table.row_values(idx)[:5]
                # Consider the priority of pairs 
                if row[2].strip():            
                    chs = row[2].split()
                    if len(chs)==1:
                        continue
                    for i in range(1,len(chs)):
                        freq = row[1] if type(row[1])==float else 1.0                
                        ch_dict[chs[i]].add((int(freq),chs[0]))
                elif row[3].strip():
                    corr_seq = row[3].strip()
                    error_seq = row[4].strip()
                    word_dict[error_seq] = corr_seq
            yield (filename, ch_dict, word_dict)
            continue
        elif filename.startswith('udn_pairs'):
            ch_dict = defaultdict(set)
            with open(path, 'r', encoding='utf8') as fp:
                for line in fp:
                    tt = line.split()
                    if int(tt[2])>10:
                        ch_dict[tt[0]].add((int(tt[2]), tt[1]))
                    
            yield (filename, ch_dict, dict())
            continue
        
        print(filename)
        
        # For 1,2,3,4
        if len(df)>0:
            df = df.dropna()
            df['idx'] = df.apply(lambda x:x['錯誤詞'].find(x['錯誤字']), axis=1)
            df['pair'] = tuple(zip(df['idx'], df['錯誤字']))
            df['noMultiErrors'] = df.apply(lambda x:x['正確詞']==x['錯誤詞'].replace(x['錯誤字'],x['正確字']), axis=1)
            
            # Remove multi-errors for the lack of right answer 
            preCnt = len(df)
            df = df[df['noMultiErrors']==True]
            postCnt = len(df)
            
            print('Original:{}\tPost:{}'.format(preCnt,postCnt))
            
            df = df.set_index('錯誤詞')
            
            # Output DICT{'error_word':'(idx, corr_ch)'}
#             df_slice = df[['pair']]
#             word_dict = df_slice.to_dict()['pair']
            word_dict = df[['正確詞']].to_dict()['正確詞']

            # output DICT{'error_ch':set(cands)}
            ch_dict = defaultdict(lambda :set())
            pairs = tuple(zip(df['錯誤字'], df['正確字']))
            for error_ch, corr_ch in pairs:
                ch_dict[error_ch].add(corr_ch)

            yield (filename, ch_dict, word_dict)


# ## Confusion Sentnece (from SIGHAN)

# 1. Bakeoff-2013 not work
# 2. sequence error not append 

# In[16]:

def extractSentence(filelist):
    for filename, path in filelist.items():
        print('== Filename: {}'.format(filename))
        
        with open(path,'r',encoding='utf8') as fp:
            soup = BeautifulSoup(fp, 'lxml')
        
        ch_dict = defaultdict(set)
        word_dict = defaultdict(set)
        seq_dict  = defaultdict(set)
        
        # Different label
        if filename.startswith('Bakeoff'):
            pass 
            ############## NOT FIX
            for idx,element in enumerate(soup.find_all('DOC')):  
                # Text
                text = dict()
                for pas in element.find('p').find_all('passage'):
                    text[pas.get('id')] = pas.string

                # Mistake
                for mistake in element.find_all('mistake'):
                    mis_id = mistake.get('id')
                    mis_loc = mistake.get('location')
                    mis_wrong = mistake.find('wrong').string.strip()
                    mis_corr  = mistake.find('correction').string.strip()
                    cur_seq = text.get(mis_id, '')

                    pairs =  [(mis_wrong,idx,x,y) for idx, (x,y) in enumerate(zip(mis_wrong, mis_corr)) if x!=y]

                    # error-corr
                    for mis_wrong,idx,error_ch,corr_ch in pairs:
                        # char-based
                        ch_dict[error_ch].add(corr_ch)

                        # word-based
                        word_dict[mis_wrong].add((idx,corr_ch))

        else:            
            for idx,element in enumerate(soup.find_all('essay')):  
                # Text
                text = dict()
                for pas in element.find('text').find_all('passage'):
                    text[pas.get('id')] = pas.string

                # Mistake
                for mistake in element.find_all('mistake'):
                    mis_id = mistake.get('id')
                    mis_loc = mistake.get('location')
                    mis_wrong = mistake.find('wrong').string.strip()
                    mis_corr  = mistake.find('correction').string.strip()
                    cur_seq = text.get(mis_id, '')

                    pairs =  [(mis_wrong,idx,x,y) for idx, (x,y) in enumerate(zip(mis_wrong, mis_corr)) if x!=y]

                    # error-corr
                    for mis_wrong,idx,error_ch,corr_ch in pairs:
                        # char-based
                        ch_dict[error_ch].add(corr_ch)

                        # word-based
                        word_dict[mis_wrong].add((idx,corr_ch))

                        # sequence-based 
                        ### Have problem with multiple errors in single word 
            #             seq_dict[cur_seq].add((int(mis_loc)-1,corr_ch))
    
        yield (filename, ch_dict, word_dict, seq_dict)


# ## The other 

# In[ ]:




# # ALL

# In[17]:

# dataroot = 'G:/UDN/training_confusion/{}/'.format
dataroot = '/home/kiwi/udn_data/training_confusion/{}/'.format


# # * Char information

# In[18]:

section_label = 'char_information'
filelist = dict((file,dataroot(section_label)+file) for file in os.listdir(dataroot(section_label)))


# In[19]:


dicBPMF = defaultdict(list)
dicPhone = defaultdict(list)
dicCangjie = defaultdict(list)
dicCangjie2char = defaultdict(list)
unihan_extract(filelist['unihan.csv'])

sound_SIGHAN = pd.read_csv(
    filelist['Bakeoff2013_CharacterSet_SimilarPronunciation.txt'], 
    sep='\t', index_col=0)
shape_SIGHAN = pd.read_csv(
    filelist['Bakeoff2013_CharacterSet_SimilarShape.txt'], \
    sep=',', index_col=0, names=['cands']).to_dict()['cands']

voc = zwtTitle_train(open(filelist['zwt.titles.txt'], encoding='utf8').readlines())

dicRadicalnum, dicRadical = radicalDic(
    open(filelist['radical.txt'], 'r', encoding='utf8').readlines())

dicFreq = bigUnihan_extract(filelist['unihan_utf8_new.csv'])


# # * Error_corr_pair

# In[20]:

section_label = 'error_corr_pair'


# In[21]:

filelist = dict((file,dataroot(section_label)+file) for file in os.listdir(dataroot(section_label)))
confusion_pairs = dict()
# confusion_pairs[special] = extractPairs_udn
for filename, ch_dict, word_dict in extractPairs(filelist):
    print('ch_dict:{}\tword_dict:{}\n'.format(len(ch_dict),len(word_dict)))
    confusion_pairs[filename] = (ch_dict,word_dict)


# # * Error_corr_sentence

# In[22]:

section_label = 'error_corr_sentence'
filelist = dict((file,dataroot(section_label)+file) for file in os.listdir(dataroot(section_label)))
unwated_file = filelist.pop('big5')

confusion_sentences = dict()
for filename, ch_dict, word_dict, seq_dict in extractSentence(filelist):
    print('ch_dict:{}\tword_dict:{}\n'.format(len(ch_dict),len(word_dict)))
    confusion_sentences[filename] = (ch_dict,word_dict)


# # * Char_probability

# In[49]:

from model.lm import LM

filename = '/home/kiwi/udn_data/training_confusion/sinica.corpus.seg.char.lm'
lm = LM(filename)


# In[51]:

lm.scoring('地')


# # Confusion_training 

# In[23]:

def sound_compare_unihan(ch_x,ch_y):
    if ch_y in sound_extract_same(ch_x):
        return 4
    elif ch_y in sound_extract_tone(ch_x):
        return 3
    elif ch_y in sound_extract_similartConsonant(ch_x, toneKeep=True):
        return 2
    elif ch_y in sound_extract_similartConsonant(ch_x, toneKeep=False):
        return 1
    else:
        return 0
    
def shape_compare_unihan(ch_x,ch_y):
    if ch_y in shape_similar(ch_x):
        return 1
    else:
        return 0


# In[58]:

def comparison4confusion(ch_chunk):
    ch_x = ch_chunk[0]
    ch_y = ch_chunk[1]
    
    log = list()
    score = 0.0
    
    # MaxScore = 36 (pair/sentence count:5)
    
    tmp = sound_compare_unihan(ch_x,ch_y)
    log.append(tmp)
    score += tmp
    
    tmp = shape_compare_unihan(ch_x,ch_y)
    log.append(tmp)
    score += tmp
    
    tmp = cangjie_compare_unihan(ch_x,ch_y)
    log.append(tmp)
    score += tmp

    tmp = sound_compare_SIGHAN(ch_x,ch_y)
    log.append(tmp)
    score += tmp

    
    tmp = shape_compare_SIGHAN(ch_x,ch_y)
    log.extend(tmp)
    score = score + tmp[0] + tmp[1]
    
    tmp = (5.0-dicFreq.get(ch_y,5))
    log.append(tmp)
    score += tmp    
    
    # log probability 
    tmp = lm.scoring(ch_x)
    log.append(tmp)
    score -= tmp
    
    # log probability 
    tmp = lm.scoring(ch_y)
    log.append(tmp)
    score -= tmp
    
    evidence = []
    for i in [confusion_pairs.items(), confusion_sentences.items()]:
        tmp = 0
        for filename, (ch_dict,_) in i:
            if ch_y in ch_dict.get(ch_x,[]):
                tmp += 1
                evidence.append(filename)
                score += 1
        
        log.append(tmp)
    log.append(evidence)
    

    return (score,log)


# In[43]:

confusion_pairs['1新編常用錯別字門診.txt'][0]['事']


# In[59]:

comparison4confusion(('事','是'))


# # Char_comparison

# In[25]:

def comparison(ch_x, ch_y):
    print('=== Comparison')
    print(ch_x, ch_y)
    '''
    相似音的處理
    sound_extract_same
    sound_extract_tone
    sound_extract_similartConsonant
    sound_extract_finalConsonant

    同音同調 同音異調 異音同調 異音異調
    '''
        
        
    print('\n=== Sound similar (from unihan)')
    print(sound_compare_unihan(ch_x,ch_y))
    
    '''
    shape_similar
    '''
    print('\n=== Shape similar (from radical)')
    print(shape_compare_unihan(ch_x,ch_y))

    '''
    cangjie_compare
    2 same cangjie code 
    1 similar cangjie code 
    0 nothing special
    '''
    print('\n=== Cangjie (from unihan)')
    print( cangjie_compare_unihan(ch_x,ch_y))
        
    '''
    SIGHAN
    '''
    print('\n=== SIGHAN Data (sound)')
    print(sound_compare_SIGHAN(ch_x, ch_y))
    

    print('\n=== SIGHAN Data (shape)')
    print(shape_compare_SIGHAN(ch_x,ch_y))
#     if t[0]==1:
#         print('Similar shape')
#     if t[1]==1:
#         print('同部首同筆畫數')
    

    '''
    Error-correct pair 
    'filename': (ch_dict,word_dict)
    '''
    print('\n=== Error-correct pair')
    for filename, (ch_dict,_) in confusion_pairs.items():
        if ch_y in ch_dict.get(ch_x,[]):
            print(filename)

    '''
    Error-correct sentence 
    'filename': (ch_dict, word_dict)
    '''
    print('\n=== Error-correct sentence')
    for filename, (ch_dict,_) in confusion_sentences.items():
        if ch_y in ch_dict.get(ch_x,[]):
            print(filename)


# In[26]:

# ch_x = '相'

# ch_y = random.choice(list(rr))

# comparison(ch_x,ch_y)

# ch_x = random.choice(sound_SIGHAN.index)
# ch_y = random.choice(sound_SIGHAN.index)

# comparison(ch_x,ch_y)


# In[64]:

def extractFeature(outputfilename, process_cnt, test=0):
    if test == 0:
        ch_n_label = set(dicBPMF.keys()).union(set(sound_SIGHAN.index))
    else:        
        ch_n_label = random.choices(list(ch_label),k=test)

    # %%timeit -n 1 -r 1

    # pool_size=multiprocessing.cpu_count()

    bigDict = defaultdict(dict)

    start_time = time.clock()
    with multiprocessing.Pool(processes=process_cnt) as pool:
        for ch_x in ch_n_label:
            ch_n_inside = list(ch_n_label)

            ch_n_inside.remove(ch_x)

            ch_chunk = [(ch_x, ch_y) for ch_y in ch_n_inside]

            scores = pool.map(comparison4confusion, ch_chunk)

            for idx,(_,ch_y) in enumerate(ch_chunk):
                if scores[idx][0]>0.0:
    #             if scores[idx] >= 0:
                    bigDict[ch_x][ch_y] = scores[idx]
            bigDict[ch_x][ch_x] = (30,[])

    with open(outputfilename, 'wb') as fp:
        pickle.dump(bigDict,fp)
    
    print(time.clock()-start_time)
    


# In[72]:

if __name__=="__main__":
    if len(sys.argv) == 3:
        extractFeature(
            outputfilename=sys.argv[1], 
            process_cnt=sys.argv[2], 
            test=5)
    else:
        sys.exit(0)

