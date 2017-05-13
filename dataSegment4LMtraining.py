
# coding: utf-8

# In[1]:

import os
import re 
import sys 
from bs4 import BeautifulSoup
import jieba

# In[32]:

def runUdnSegment(dataroot, outputfile, action, unwanted_ptn):
    '''Extract UDN news, and ouput char-level segement file 
    
    Need to revise pattern regular experission 
    
    Args:
        dataroot (str): the position of UDN news (recursive)
        outputfile (str): the position of output file 
        action (dict): parameter for seperate and segmentation-level 
        unwanted_ptn (rgex): pattern for kicking unwanted character 
    Return: 
        None
    '''
    
    sep_method = action.get('sep_method', 'comma')
    seg_level = action.get('seg_level', 'char')
    
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
                    soup = BeautifulSoup(data, 'lxml')
                    
                content = contentExtract(soup)
                seqs = seperateSeq(content, sep_method)
                seqs_filter = filterSeq(seqs, unwanted_ptn)
                seg_string = transformSeq(seqs_filter, seg_level)
                
                with open(outputfile, 'a', encoding='utf8') as wp:
                    wp.write(seg_string+'\n')


# In[15]:

def contentExtract(soup):
    '''Extract the string content of website 
    Args:
        soup (Beatutifulsoup): website 
    Return:
        output (str): the string content of website
    '''
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


# In[16]:

def seperateSeq(content, sep_method):
    '''Seperate string into sub-sentence
    Args:
        content (str): website content 
        sep_method (str): 'original' or 'comma'
    Return:
        output (list): sub-sentence
    '''
    output = []
    if sep_method == 'comma':
        pattern = re.compile('[，。！？]')
        content = ''.join(content.split('\n'))

        pre_idx=0
        for idx, ch in enumerate(content):
            if pattern.search(ch):
                tmp = content[pre_idx:idx+1]
                output.append(tmp)
                pre_idx = idx+1

    #     print(pre_idx, len(seq))
        if pre_idx<len(content):
            tmp = content[pre_idx:len(content)]
            output.append(tmp)
    elif sep_method:
        output = content.split('\n')
    
    return output 


# In[17]:

def filterSeq(lst, pattern):
    '''Filter unwanted sequence based on pattern
    Args:
        lst (list): the list of website seperated content 
        pattern (rgex): the pattern we don't want 
    Return:
        output (list): list after filter
    '''
    
    output = [seq for seq in lst if not pattern.search(seq)]
    
    if not output:
        return list()
    
    if output[0].find('】'): 
        _ = output.pop(0)
    
    return output 


# In[18]:

def transformSeq(seqs, seg_level):
    '''filter the line existed Unwatned pattern, and seperate the char with "space"
    Args:
        seqs (list): list from website 
        seg_level: which lm-level ('word' OR 'char)
    Return:
        output (str): string which have been seperated by 'space'
    '''
    output = list()
    if seg_level == 'word':
        for seq in seqs:
            segs = jieba.cut(seq)
            output.append(' '.join(segs))            
    elif seg_level == 'char':
        for seq in seqs:
            output.append(' '.join(seq))
    return '\n'.join(output)


# In[19]:

def runSinica(inputfile, outputfile):
    with open(inputfile, 'r', encoding='utf8') as fp,    open(outputfile, 'w', encoding='utf8') as wp:
        for line in fp:
            data = line.strip('\n').split(' ')
            output = []
            for item in data:
                tmp = item.split('|')
                if len(tmp)==2:
                    output.append(tmp[0])

            wp.write(' '.join(output))
            wp.write('\n')
    #         print(' '.join(output))
        


# In[33]:

if __name__=='__main__':
    ptn = re.compile('[A-Za-z0-9.\s]')
    par = {
        'sep_method':'comma'
        , 'seg_level':'char'}

    data_root = sys.argv[1]
    output_file = sys.argv[2]
    
    runUdnSegment(dataroot=data_root, outputfile=output_file,
                 action=par, unwanted_ptn=ptn)
    

#     dataroot = '/home/kiwi/Documents/udn_data/Files/'
#     runCharSegment(dataroot,outputfile,pattern)
#     runWordSegment(dataroot,'./lm_data/seg_all.txt')
#     inputfile = '/home/kiwi/udn_data/training/sinica.corpus.txt'
#     runSinica(inputfile,'./lm_data/sinica_word.txt')


# In[ ]:



