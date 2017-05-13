
# coding: utf-8

# In[1]:

import os
from bs4 import BeautifulSoup
from collections import namedtuple
import re


# ### Class Name
# * 1  Add
# * 3  Delete
# * 9  Hightlight

# In[2]:

def special_debug():
    output = []
    for batch in soup.find_all('p'):
        for section in batch.contents:
            if section.name == None:

                print(0, '==\t', section.string)
            elif section.name == 'font':
                if section['class'][0] == '1':
                    print(1, 'Add\t', section.string)
                elif section['class'][0] == '2':
                    print(2, section.string)
                elif section['class'][0] == '3':
                    print(3, section.string)
    #                 if pre.action=='3':
    #                     pre.string += section.string
    #                 else:
    #                     output.append(pre)
    #                     pre = tag_attri('3',section.string)

                else:
                    print('====\nWrong\n%s\n====\n' % (section.string))
            else:
                print('wrong wrong')


# In[3]:

def seperateSeq(seq):
    pattern = re.compile('[，。！？]')

    pre_idx = 0
    output = []
    for idx, ch in enumerate(seq):
        if pattern.search(ch):
            tmp = seq[pre_idx:idx + 1]
            output.append(tmp)
            pre_idx = idx + 1

#     print(pre_idx, len(seq))
    if pre_idx < len(seq):
        tmp = seq[pre_idx:len(seq)]
        output.append(tmp)

    return output


# In[4]:

def extract(soup):
    tag_attri = namedtuple('tag_attri', 'action, string')
    pre = tag_attri('-5', '')
    output = []
    for batch in soup.find_all('p'):
        for section in batch.contents:
            if section.name == None:
                cur_action = '0'
            elif section.name == 'font':
                cur_action = section['class'][0]
            else:
                continue
                print(section)
                cur_action = '-1'
                print('wrong wrong')
#                 return -1

            if section.string == None:
                continue
#             print(section.string)
            insert = section.string.replace('\n', '')
#             print(insert)
            if pre.action == cur_action:
                i = pre.string + insert
                pre = tag_attri(cur_action, i)
            else:
                output.append(pre)
                pre = tag_attri(cur_action, insert)
#     print(pre.string, section.string)
#     if pre.string!=section.string:
    output.append(pre)

    # Match action and character
    seq_idx = []
    seq_ch = ''
    for term in output:
        #         if term.string==None:
        #             continue
        seq_idx = seq_idx + [int(term.action) for _ in range(len(term.string))]
        seq_ch = seq_ch + term.string

    # Seperate sequence by comma
    seq_seperate = seperateSeq(seq_ch)

    total_length = 0
    batch = []
    for sub in seq_seperate:
        check = seq_idx[total_length + len(sub) - 1]

        # =====================
        # Discard those data ends with deleted comma
        # =====================

        if check != 3:
            seq_idx[total_length + len(sub) - 1] = 'x'
            batch.append((seq_idx[total_length:total_length + len(sub)], sub))

        total_length += len(sub)

    return batch


# In[5]:

# Skip the first sequence for the name of reporter
# Situation:
# preprocess: unnecessary comma (abandon this sequence or delete comma only)
# 1. 0 only- original content
# 2. 0/1 only- add some word
# 3. 0/3 only- delete some word
# 4. 1 only- add content
# 5. 3 only- delete content
# 6. 0/1/3 only- Special situation:
#  - 1/3 nearby: replace
#  - 1/3 not nearby: add/delete some word with different purpose

def sepCase(batch):
    trace = 0
#     pattern = re.compile('「」【】；：')
#     ptn = pattern.findall(cur_seq)
#     if ptn == ['「', '」'] or ptn == ['【', '】']:
#         continue
    for cur_idx, cur_seq in batch:
        tag = set(cur_idx[:-1])
        idx_all = set(range(len(cur_idx)))

        # 1. 0 only- original content
        if tag == set([0]):
            if trace:
                print('1', cur_idx, cur_seq)

            out = cur_seq
            sep = ''

            yield('case1', cur_seq, cur_idx)

        # 2. 0/1 only- add some word
        elif tag == set([0, 1]):
            if trace:
                print('2', cur_idx, cur_seq)
            yield('case2', cur_seq, cur_idx)

        # 3. 0/3 only- delete some word
        elif tag == set([0, 3]):
            if trace:
                print('3', cur_idx, cur_seq)
            yield('case3', cur_seq, cur_idx)

        # 4. 1 only- add content
        elif tag == set([1]):
            if trace:
                print('4', cur_idx, cur_seq)
            yield('case4', cur_seq, cur_idx)

        # 5. 3 only- delete content
        elif tag == set([3]):
            if trace:
                print('5', cur_idx, cur_seq)

            yield('case5', cur_seq, cur_idx)

        # 6. 0/1/3 only- Special situation:
        #  - 1/3 nearby: replace
        #  - 1/3 not nearby: add/delete some word with different purpose
        elif tag == set([0, 1, 3]):
            if trace:
                print('6', cur_idx, cur_seq)

            (c6_seq, c6_cor) = case6(cur_idx, cur_seq)

            if c6_cor != 0:
                yield ('case6', c6_seq, c6_cor)


# In[12]:

def case6(cur_idx, cur_seq, trace=0):
    out_idx = list(cur_idx)
    out_seq = list(cur_seq)
    corr = []

    _idxs = ''.join(str(x) for x in cur_idx)
    pattern = re.compile('(0130)|(0310)')
    ptn = pattern.finditer(_idxs)

    # =====
    # Remove comma replacement
    # =====
    kick_ptn = re.compile('[『「」（：）()』／；●】【~～〈〉《》＆\-、★\—\'％%‧○…■\s]')
    kick_ptn2 = re.compile('[0-9A-Za-z]')

    # 5: kick tag
    change = 0
    for p in ptn:
        _start = p.start()
        if p.group() == '0130':
            if out_seq[_start + 1] == out_seq[_start + 2]:
                continue

            change = 1
            corr.append(out_seq[_start + 1])
            out_seq[_start + 1] = ''
            out_idx[_start + 1] = 5

            out_idx[_start + 2] = 4

        elif p.group() == '0310':
            if out_seq[_start + 1] == out_seq[_start + 2]:
                continue

            change = 1
            out_idx[_start + 1] = 4

            corr.append(out_seq[_start + 2])
            out_seq[_start + 2] = ''
            out_idx[_start + 2] = 5

        if kick_ptn.search(corr[-1]) != None or kick_ptn2.search(corr[-1]) != None or corr[-1] == '\xa0':
            return (cur_seq, 0)

    # =============
    # Don't handle the situation over the pattern
    # =============
    if change == 0:
        return (cur_seq, 0)

    # Remove character with tag-3 delete
    _idxs = ''.join(str(x) for x in out_idx)
    for p in re.finditer(r'3', _idxs):
        start = p.start()
        out_seq[start] = ''
        out_idx[start] = 5
    out_idx = [i for i in out_idx if i != 5]
    out_seq = ''.join(out_seq)

    # Error index start from 1
    # KICK unwanted error-correction pair based on re.pattern
    # Create list(tuple(position, correct))
    _idxs = ''.join(str(x) for x in out_idx)
    out_cor = []
    if len(out_idx) == len(out_seq):
        for idx, p in enumerate(re.finditer(r'4', _idxs)):
            #             print(out_seq[p.start()], corr[idx])
            out_cor.append((p.start() + 1, corr[idx]))

    return (out_seq, out_cor)


# In[14]:

def document(inputfile, outputname):
    (ip_cor_name, ip_error_name, gt_name) = outputname
    (inputname, document_id) = inputfile

    with open(inputname, 'rb') as fp:
        data = fp.read().decode('big5-hkscs', 'ignore')
        soup = BeautifulSoup(data)

#     with open(inputname, 'r', encoding='utf8') as fp:
#         soup = BeautifulSoup(fp)

    batch = extract(soup)

# 1. 0 only- original content
# 2. 0/1 only- add some word
# 3. 0/3 only- delete some word
# 4. 1 only- add content
# 5. 3 only- delete content
# 6. 0/1/3 only- Special situation:
#  - 1/3 nearby: replace
#  - 1/3 not nearby: add/delete some word with different purpose
    kick_ptn = re.compile(r'[0-9A-Za-z]')

    with open(ip_cor_name, 'a', encoding='utf8') as icp,     open(gt_name, 'a', encoding='utf8') as gtp,    open(ip_error_name, 'a', encoding='utf8') as iep:
        for idx, (c, seq, special) in enumerate(sepCase(batch)):
            sentence_id = '%s-%d' % (document_id, idx)

            if c == 'case6' and kick_ptn.search(seq) == None:
                iep.write('%s, %s\n' % (sentence_id, seq))
                gtp.write('%s' % sentence_id)
                for item in special:
                    gtp.write(', %d, %s, %s' %
                              (item[0], seq[item[0] - 1], item[1]))
                gtp.write('\n')

            elif c == 'case1' or c == 'case2' or c == 'case4':
                icp.write('%s, %s\n' % (sentence_id, seq))


# In[8]:

def docu_debug(inputname):
    #     with open(inputname, 'rb') as fp:
    #         data = fp.read().decode('big5-hkscs', 'ignore')
    #         soup = BeautifulSoup(data)

    with open(inputname, 'r', encoding='utf8') as fp:
        soup = BeautifulSoup(fp)

    batch = extract(soup)
#     return batch
    for idx, (c, seq, special) in enumerate(sepCase(batch)):
        if c == 'case6':
            print(c, seq, special)


# In[15]:

if __name__ == '__main__':

    token = 'withError'
    dataroot = 'G:/UDN/Files/'

    ip_cor_name = './extractUDN/{}_correct.txt'.format(token)
    ip_err_name = './extractUDN/{}_input.txt'.format(token)
    gt_name = './extractUDN/{}_groundtruth.txt'.format(token)
    error_file = './extractUDN/{}_errorfile.txt'.format(token)
    outputname = (ip_cor_name, ip_err_name, gt_name)

    if os.path.exists(error_file):
        print('Clean %s' % (error_file))
        os.remove(error_file)

    for file in outputname:
        if os.path.exists(file):
            print('Clean %s' % (file))
            os.remove(file)

    log_file = './extractUDN_log.txt'
    sfp = open(log_file, 'w', encoding='utf8')
    efp = open(error_file, 'w', encoding='utf8')

    for dirPath, dirName, filelist in os.walk(dataroot, topdown=False):
        if not(re.search(r'/201......TEXT$', dirPath)):
            #         if not(re.search(r'8/201608..$', dirPath)):
            continue

        print(dirPath)
        fileSet = set()
        for file in filelist:
            ptn = re.search(r'-.....-...', file)
            if ptn:
                if ptn.group() not in fileSet:
                    fileSet.add(ptn.group())
#                     inputfile = ('%s/%s' %(dirPath, file), dirPath[-4:]+ptn.group())
                    inputfile = ('%s/%s' % (dirPath, file),
                                 dirPath[-9:-5] + '-' + file[:ptn.end()])
                    sfp.write(inputfile[0] + '\n')

                    try:
                        document(inputfile, outputname)
                    except:
                        efp.write(inputfile[0] + '\n')

            else:
                with open(error_file, 'a', encoding='utf8') as fp:
                    fp.write('%s/%s\tptn.group()\n' % (dirPath, file))

    sfp.close()
    efp.close()
    print('--- FINISHED ---')
