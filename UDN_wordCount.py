
# coding: utf-8

# In[1]:

import pickle
from collections import defaultdict

import pandas as pd


# In[ ]:

# if False:
#     cnt_char = defaultdict(int)
#     with open(seqfilename, 'r', encoding='utf8') as fp:
#         for line in fp:
#             lst = line.strip().split()
#             for ch in lst:
#                 cnt_char[ch] += 1

#     pickle.dump(file=open('UDN_charCount.pkl', 'wb'), obj=cnt_char)

#     sort_cnt_char = sorted(cnt_char.items(), key=lambda x:(x[1],len(x[0])), reverse=True )

#     with open('UDN_charCount.txt', 'w', encoding='utf8') as wp:
#         for c, cnt in sort_cnt_char:
#             wp.write('{},{}\n'.format(c,cnt))


# In[ ]:

def count_char(seqfilename):
    cnt_char = defaultdict(int)
    with open(seqfilename, 'r', encoding='utf8') as fp:
        for line in fp:
            lst = line.strip().split()
            for ch in lst:
                cnt_char[ch] += 1
    
    return cnt_char


# In[70]:

seqfilename = '/home/kiwi/udn_data/UDN.sentence.char.txt'
# seqfilename = 'C:/Users/newslab/Desktop/UDN.sentence.char.txt'
# seqfilename = 'G:/UDN/lm_data/UDN.sentence.char.txt'


# In[ ]:

# char_frequency
cnt_char = count_char(seqfilename)


# In[2]:

preErrorfilename = './extractUDN_new/all/all_preError.csv'
errorPostfilename = './extractUDN_new/all/all_errorPost.csv'


# In[5]:

df_preError = pd.read_csv(preErrorfilename, sep='\t')
df_postError = pd.read_csv(errorPostfilename, sep='\t')


# In[65]:

pre_words = []
post_words = []
words = set()
for _, row in df_preError.iterrows():
    errorword = '{}{}'.format(row['pre'],row['error'])
    corrword = '{}{}'.format(row['pre'], row['corr'])    
    pre_words.append((errorword, corrword))
    words.update([errorword, corrword])
    
for _, row in df_postError.iterrows():
    errorword = '{}{}'.format(row['error'], row['post'])
    corrword = '{}{}'.format(row['corr'], row['post'])
    post_words.append((errorword, corrword))
    words.update([errorword, corrword])


# In[93]:

def batch(chunk):
    return 1 if chunk in words else 0


# In[95]:

words_count = defaultdict(int)

with open(seqfilename, 'r', encoding='utf8') as fp, multiprocessing.Pool(processes=2) as pool:
#     for line in fp:
#     if True:
    for i in range(3000):
        line = fp.readline()
        line_str = ''.join(line.strip().split())
        
        search_lst = [''.join(search) for search in zip(line_str[:-1:], line_str[1::])]
        tag = pool.map(batch, search_lst)
        
        tag_word = [search_lst[idx] for idx, t in enumerate(tag) if t==1]
        
        for w in tag_word:
            words_count[w] += 1

pickle.dump(file=open('UDN_wordscount.pkl','wb'), obj=words_count)


# In[6]:

# search_str = \
# '今天 昨天 污染 汙染 佈局 布局 臺灣 台灣 秘書 祕書 記錄 紀錄 越來 愈來 志工 義工 週邊 周邊 週期 周期 佔率 占率\
#  規劃 規畫 計劃 計畫 部份 部分 比例 比率 發佈 發布 提昇 提升 身份 身分 每週 每周 上週 上周 市佔 市占 公佈 公布'

# search_token = dict()
# for i in search_str.split():
#     search_token[i] = 0
    
# words_count = defaultdict(int)

# with open(seqfilename, 'r', encoding='utf8') as fp:
#     for line in fp:
#         line_str = ''.join(line.strip().split())
#         for search in zip(line_str[:-1:],line_str[1::]):
#             s = ''.join(search)
#             if s in search_token:
#                 search_token[s] += 1

# with open('errorpairCount.csv', 'w', encoding='utf8') as wp:
#     for p, cnt in search_token.items():
#         wp.write('{},{}\n'.format(p,cnt))

