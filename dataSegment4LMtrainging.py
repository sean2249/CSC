
# coding: utf-8

# In[3]:

import os
import re 
import jieba
from bs4 import BeautifulSoup


# In[4]:

def runCharSegment(dataroot, outputfile):
    if os.path.exists(outputfile):
        print('Clean %s' %(outputfile))
        os.remove(outputfile)
    
    pattern = re.compile('TEXT')
    for dirPath, dirName, filelist in os.walk(dataroot, topdown=False):
        if pattern.search(dirPath):        
            print(dirPath)
            for file in filelist:
                inputfile = dirPath+'/'+file
    #             print(inputfile)
                with open(inputfile, 'rb') as fp:
                    data = fp.read().decode('big5-hkscs', 'ignore')
                    soup = BeautifulSoup(data)
                content = contentExtract(soup)
                seperateSeq = seperate(content)

                with open(outputfile, 'a', encoding='utf8') as wp:
                    wp.write(seperateSeq+'\n')


# In[56]:

def runWordSegment(dataroot, outputfile):
    # Turn all the number into 0 for better nubmer count
    
    if os.path.exists(outputfile):
        print('Clean %s' %(outputfile))
        os.remove(outputfile)
    
    pattern = re.compile('TEXT')
    for dirPath, dirName, filelist in os.walk(dataroot, topdown=False):
        if pattern.search(dirPath):
            print(dirPath)
            for file in filelist:
                inputfile = dirPath+'/'+file
                with open(inputfile, 'rb') as fp:
                    data = fp.read().decode('big5-hkscs', 'ignore')
                    soup = BeautifulSoup(data)
                content = contentExtract(soup)
                seg = wordSegment(content)
                
                with open(outputfile, 'a', encoding='utf8') as wp:
                    wp.write(seg+'\n')


# In[11]:

def seperateSeq(seq):
    pattern = re.compile('[，。！？]')
    
    pre_idx=0
    output = []
    for idx, ch in enumerate(seq):
        if pattern.search(ch):
            tmp = seq[pre_idx:idx+1]
            output.append(tmp)
            pre_idx = idx+1
    
#     print(pre_idx, len(seq))
    if pre_idx<len(seq):
        tmp = seq[pre_idx:len(seq)]
        output.append(tmp)
        
    return output 


# In[54]:

def wordSegment(content):
    segs = list(jieba.cut(content))
    
    for idx, seg in enumerate(segs):
        if re.search('[0-9]', seg):
            segs[idx] = '0'
    
    return ' '.join(segs)


# In[7]:

def extractStrOnly(element):
    
    flag = element.name
    output = ''    
    res = []
    if flag!=None:
        for item in element.contents:
            output += extractStrOnly(item)
    else:
        return element.string
    
    return output


# In[8]:

def contentExtract(soup):
    output = ''
    for pTxt in soup.find_all('p'):
        res = ''
        for tag_c in pTxt.contents:
            try:
                if tag_c.get('class')==1:
                    res = res+tag_c.string
            except:
                res = res + tag_c
        res = res.strip('.\f\n\r\t\v')
        if len(res)==0:
            continue
        output = output + res+'\n'
#         print(output)
    return output


# In[9]:

def seperate(content):
    seqs = content.split('\n')
    output = []
    pattern = re.compile('[A-Za-z0-9.\s]')
    for seq in seqs:
        if len(seq)==0:
            continue
        
        tmpSeq = seq[0]
        preflag = 0
        for ch in seq[1:]:
            if pattern.search(ch):
                continue
#                 if preflag ==0:
#                     tmpSeq = tmpSeq+' '
#                     preflag = 1
#                 tmpSeq = tmpSeq+ch
                
            else:                
                preflag=0
                tmpSeq = tmpSeq + ' '+ ch
        output.append(tmpSeq)
        
    return '\n'.join(output)


# In[57]:

if __name__=='__main__':
#     dataroot = '/home/kiwi/Documents/udn_data/Files/'
    dataroot = 'G:/UDN/Files/'
#     outputfile = 'all_content.txt'
#     runCharSegment(dataroot,outputfile)
    runWordSegment(dataroot,'./lm_data/seg_all.txt')


# In[ ]:



