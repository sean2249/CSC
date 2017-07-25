
# coding: utf-8

# In[1]:

import os
from bs4 import BeautifulSoup
from collections import namedtuple, defaultdict
import re


# ### Class Name 
# * 1  Add 
# * 3  Delete
# * 9  Hightlight

# In[2]:

def seperateSeq(seq):
    '''Seperate passage into small sentences
    Args:
        seq (string): string of passage 
    Return:
        output (list): list of sentences            
    '''
    pattern = re.compile('[，。！？]')
    
    pre_idx=0
    output = []
    for idx, ch in enumerate(seq):
        if pattern.search(ch):
            tmp = seq[pre_idx:idx+1]
            output.append(tmp)
            pre_idx = idx+1
    
    if pre_idx<len(seq):
        tmp = seq[pre_idx:len(seq)]
        output.append(tmp)
        
    return output 


# In[3]:

# filename = '/home/kiwi/udn_data/Files/20160618/TEXT/113-06684-003B_20160618171735_03684.TXT'

# with open(filename, 'rb') as fp:
#     data = fp.read().decode('big5-hkscs', 'ignore')
#     soup = BeautifulSoup(data)


# In[4]:

def extract(soup):    
    '''Extract action/content from soup, seperated with comma.    
    
    action: 
        * 0- default
        * 1- add
        * 3- delete
        * 'x'- end of sentence
    
    Args:
        soup (BeautifulSoup.soup): the soup element of document 
        
    Return:
        output (int,str): tuple (action of characters, characters)    
        
    '''
    tag_attri = namedtuple('tag_attri','action, string')
    pre = tag_attri('-5', '')
    output = []
    # Extract section string corrsponding action 
    for batch in soup.find_all('p'):
        for section in batch.contents:           
            if section.name==None:
                cur_action = '0'
            elif section.name=='font':
                cur_action = section['class'][0]
            else:
                continue
                print(section)
                cur_action = '-1'
                print('wrong wrong')
            
            if section.string==None:
                continue
            insert = section.string.replace('\n','')
            if pre.action == cur_action:                
                i = pre.string+insert
                pre = tag_attri(cur_action, i)                                
            else:
                output.append(pre)
                pre = tag_attri(cur_action, insert)                   
    output.append(pre)

    # Match action and character 
    seq_idx = []
    seq_ch = ''
    for term in output:
        seq_idx = seq_idx + [int(term.action) for _ in range(len(term.string))]
        seq_ch = seq_ch + term.string

    # Seperate sequence by comma 
    seq_seperate = seperateSeq(seq_ch)
    
    total_length = 0
    batch = []
    for sub in seq_seperate:
        check = seq_idx[total_length+len(sub)-1]
        
        # =====================
        # Discard those data ends with deleted action of comma 
        # =====================

        if check != 3:
            seq_idx[total_length+len(sub)-1] = 'x'
            batch.append((seq_idx[total_length:total_length+len(sub)], sub))
        
        total_length += len(sub)    
    
    return batch


# In[5]:

def sepCase(batch, show=0):
    '''Classify the action/content into 6 cases
    
    Args:
        batch (list(int), str): datatype from the function of 'extract'
    
    Yields:
        return different case
    
    '''    
    for cur_idx, cur_seq in batch:
        tag = set(cur_idx[:-1])

        # 1. 0 only- original content
        if tag == set([0]):
            yield('case1', cur_seq, cur_idx)

        # 2. 0/1 only- add some word
        elif tag == set([0, 1]):
            yield('case2', cur_seq, cur_idx)

        # 3. 0/3 only- delete some word
        elif tag == set([0, 3]):
            yield('case3', cur_seq, cur_idx)

        # 4. 1 only- add content
        elif tag == set([1]):
            yield('case4', cur_seq, cur_idx)


        # 5. 3 only- delete content
        elif tag == set([3]):
            yield('case5', cur_seq, cur_idx)

        # 6. 0/1/3 only- Special situation: 
        #  - 1/3 nearby: replace 
        #  - 1/3 not nearby: add/delete some word with different purpose
        elif tag == set([0,1,3]):
            yield('case6', cur_seq, cur_idx)                


# In[6]:

def case6Process(cur_seq, cur_idx, 
                 select_ptn='(0130)|(0310)', 
                 kick_ptn='[『「」（：）()』／；●】【~～〈〉《》＆\-、★\—\'％%‧○…■\s]', 
                 show=0):
    
#     def tagChange(in_idx, in_seq):
    
    
    out_idx = list(cur_idx)
    out_seq = list(cur_seq)
    corr = []

    _idxs = ''.join(str(x) for x in cur_idx)
    pattern = re.compile(select_ptn)
    ptn = pattern.finditer(_idxs)
    
    # =====
    # Remove unwanted symbols
    # =====
    # NOOOO USE
#     KICKPTN = re.compile(kick_ptn) 
#     KICKPTN2 = re.compile('[0-9A-Za-z]')
    
    #####
    change = 0
    for p in ptn:
        _start = p.start()
        if out_seq[_start+1] == out_seq[_start+2]: 
            continue
    
        # 4- the error(keep) char
        # 5- the correct(deleted) char
        if p.group() == '0130':                    
            change = 1
            out_idx[_start+1] = 5
            out_idx[_start+2] = 4
            corr.append(out_seq[_start+1])
            out_seq[_start+1] = ''
        elif p.group() == '0310':            
            change = 1
            out_idx[_start+1] = 4
            out_idx[_start+2] = 5            
            corr.append(out_seq[_start+2])
            out_seq[_start+2] = ''
        if corr[-1]=='\xa0':
            return (cur_seq, -1)            
    if change == 0:
        return (cur_seq, -1)
    
    # Remove character with tag-3 delete
    _idxs = ''.join(str(x) for x in out_idx)
    for p in re.finditer(r'3', _idxs):
        start = p.start()
        out_seq[start] = ''
        out_idx[start] = 5
    out_idx = [i for i in out_idx if i!=5]
    out_seq = ''.join(out_seq)
    
    # Error index start from 1
    # Create list(tuple(position, correct))
    _idxs = ''.join(str(x) for x in out_idx)
    out_cor = []
    if len(out_idx) == len(out_seq):
        for idx, p in enumerate(re.finditer(r'4', _idxs)):
            out_cor.append((p.start()+1, corr[idx]))
            
    return (out_seq, out_cor)


# In[7]:

def document(inputname, document_id, kick_ptn='', encoding='big5'):
    '''Read file and output into file
    Args:
        inputfile (list): [the filename of website, the token of filename]
        outputname (str): the filename for testdata
        XXX outputname (list): [the filename for testdata file, all correct sequence]
        kick_ptn (str): the pattern which haves to kick off 
        encoding (str): the encoding of file (big5/utf8)
    
    Return:
         1  success
        -1  failed
    '''
    
#     print(inputname)
    

    if encoding == 'big5':
        with open(inputname, 'rb') as fp:
            data = fp.read().decode('big5-hkscs', 'ignore')
            soup = BeautifulSoup(data)
    elif encoding == 'utf8': 
        with open(inputname, 'r', encoding='utf8') as fp:
            soup = BeautifulSoup(fp)           
    else:
        print('wrong encoding parameters')
        return -1
        
    batch = extract(soup)    
    
    output = dict()
    for idx, case in enumerate(sepCase(batch)):
        sentence_id = '{}-{}'.format(document_id, idx)
        if case[0] == 'case6':
            seq, seq_info = case6Process(case[1], case[2])
            if seq_info != -1:
                try:
                    error_info = '|||'.join(['{}|||{}'.format(idx,ch) for idx, ch in seq_info])
                except:
                    print(seq, seq_info)
                    return -1
                output[(seq, error_info)] = sentence_id
    
    return output


# In[10]:

def create_rawdata(dataroot, output_root, kick_ptn):
    if not os.path.isdir(output_root):
        os.mkdir(output_root)
        
    seqfile = os.path.join(output_root, 'all_seq.txt')
    groundtruthfile = os.path.join(output_root, 'all_gt.txt')
    logfile = os.path.join(output_root, 'log.txt')
    badfile = os.path.join(output_root, 'bad_log.txt')
    
    seqfp = open(seqfile, 'w', encoding='utf8')
    gtfp = open(groundtruthfile, 'w', encoding='utf8')
    logfp = open(logfile, 'w', encoding='utf8')
    badfp = open(badfile, 'w', encoding='utf8')    
    
    idx = 0
    totalfilecnt = 0
    for dirpath, _, filelist in os.walk(dataroot, topdown=False):
        if not(re.search(r'/201...../TEXT$', dirpath)):
    #         if not(re.search(r'8/201608..$', dirPath)):
            continue

        print(dirpath)

        file_dict = defaultdict(list)
        file_set = set()
        totalfilecnt += len(filelist)
        for filename in filelist:
            ptn = re.search(r'^.*-.....-...', filename)
            filepath = os.path.join(dirpath, filename)
            if not ptn:
                badfp.write('{}\n'.format(filepath))
                continue
            if ptn.group() in file_set:
                file_dict[ptn.group()].append((filename, filepath))
            else:
                file_set.add(ptn.group())
                file_dict[ptn.group()].append((filename, filepath))            

        for filetoken, filelists in file_dict.items():
            document_seqs = dict()
            for (filename, filepath) in filelists:
                logfp.write('{}\n'.format(filepath))
                cur_seqs = document(filepath, os.path.splitext(filename)[0], KICK_PTN, 'big5')
                document_seqs.update(cur_seqs)

            for section, docu_id in document_seqs.items():
                seqfp.write('{}|||{}\n'.format(docu_id, section[0]))
                gtfp.write('{}|||{}\n'.format(docu_id, section[1]))

    logfp.write('========\nTotal file = {}\n======='.format(totalfilecnt))
    print('== Finish raw data ===')
    seqfp.close()
    gtfp.close()
    logfp.close()
    badfp.close()


# In[11]:

if __name__ == '__main__':
    dataroot = '/home/kiwi/udn_data/Files/'
    outputroot = './extractUDN_new/rawdata'
    KICK_PTN = ''

    create_rawdata(dataroot, outputroot ,KICK_PTN)
    


# In[ ]:



