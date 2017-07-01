
# coding: utf-8

# In[281]:

import os
import re
from collections import defaultdict
import argparse


# In[101]:

def extractErrors(seq):
    for pair in re.finditer(r'.\(.\)', seq):
        err_ch, cor_ch = pair.group()[0], pair.group()[2]
        yield err_ch, cor_ch


# In[211]:

def dataAction(seq, action):
    global testData_output, testData_output, error_type, discard
    
    if action == 'append':
        errors = []
        for (err_ch, cor_ch) in extractErrors(seq):
            if cor_ch > err_ch:
                pair = (err_ch, cor_ch) 
                error_type[pair][0] += 1
            else:
                pair = (cor_ch, err_ch)
                error_type[pair][1] += 1

            testData_pair[pair].add(seq)
            errors.append((err_ch, cor_ch))    

        testData_output[seq] = errors
    
    elif action == 'remove':
        for (err_ch, cor_ch) in extractErrors(seq):
            if cor_ch > err_ch:
                pair = (err_ch, cor_ch) 
                error_type[pair][0] -= 1
            else:
                pair = (cor_ch, err_ch)
                error_type[pair][1] -= 1
            if seq in testData_pair[pair]:
                testData_pair[pair].remove(seq)
            

        testData_output.pop(seq)        
        discard.append(seq)


# In[190]:

def loadData(labelFile):            
    global discard
    confused = []
    uniseq = dict()
    with open(labelFile, 'r', encoding='utf8') as fp:
        for line in fp:
            line = line.strip()        
            label, seq = line[0], line[1:]

            if seq in uniseq:            
                if label != uniseq.get(seq, ''):
                    confused.append(seq)                
                    if uniseq.get(seq,'E') == 'S':                    
                        dataAction(seq, 'remove')
            elif label == 'S':                        
                dataAction(seq, 'append')

            elif label == 'E':
                discard.append(seq)

            uniseq[seq] = label    
        
    return confused


# In[175]:

def confusedLabelClassify(confused):
    
    # Confused label classify 
    ####
    fid = open(UDN_benchmark_log, 'a', encoding='utf8')
    fid.write('## Conufsed Label Classify \n')
    #####

    for seq in confused:
        while(1):
            ans = input('{} want to add? y;/n\'\t'.format(seq)).lower()
            if ans == ';':
                dataAction(seq, 'append')   
                fid.write('* Append {}\n'.format(seq))
                break
            elif ans == '\'':
                fid.write('* Remove {}\n'.format(seq))
                break


# In[274]:

# Mutual errors case

def mutualCase():
    ####
    fid = open(UDN_benchmark_log, 'a', encoding='utf8')
    fid.write('## Mutual Error Case \n')
    #####
    
    mutual_case = [(pair,count) for pair, count in error_type.items()
                       if count[0]!= 0 and count[1]!=0]

    for pair, count in mutual_case:    
        allSeqs = list(testData_pair[pair])
        print('==========')
        print('\n'.join(allSeqs))
        print('{}->{} {}'.format(pair[0],pair[1],count[0]))
        print('{}->{} {}'.format(pair[1],pair[0],count[1]))    

        while(1):
            action = input('Want to process? Y;/N\'\t').lower()
            if action == ';':
                tag = 1 
                fid.write('### Process {}\n'.format(pair, count))
                break
            elif action == '\'':
                tag = 0
                fid.write('### Kick {}\n'.format(pair, count))
                break

        fid.write('{}<br>\n'.format(input('Comment: ')))

        if tag == 1:
            for seq in allSeqs:
                while(1):
                    action = input('{} Store? Y;/N\'\t'.format(seq)).lower()
                    if action == '\'':
                        dataAction(seq, 'remove')
                        fid.write('    * Remove {}\n'.format(seq))
                        break
                    elif action == ';':
                        break
        else:
            for seq in allSeqs:
                dataAction(seq, 'remove')   

    if not fid.closed: fid.close()


# In[273]:

def rankPair():
    # Ranking pair
    # testData_pair = defaultdict(set)
    # testData_output = dict()
    # error_type = defaultdict(lambda :[0,0])

    # 5 異體字，單純的相似音相似形
    # 4 要考慮到語意
    # 3 習慣用法

    #####
    fid = open(UDN_benchmark_log, 'a', encoding='utf8')
    fid.write('## Mutual Error Case \n')
    #####

    pairRanking = dict()
    items = testData_pair.items()
    for pair, seqs in items:
        if len(seqs)==0: continue
        print('======')
        print('\n'.join(list(seqs)[:5]))

        while(1):
            action = input('Rank {}-{} 0(Remove)-5(Good)\t'.format(pair[0],pair[1]))
            if not action.isdigit():
                continue
            if int(action) == 0:
                for s in list(seqs):
                    dataAction(s, 'remove')
                break
            elif int(action) <= 5:
                pairRanking[pair] = action
                break      

        action = input('Need comment? Y; ')
        if action == ';':
            fid.write('* {}- {}\n'.format(pair, input('comment: ')))
            
            
    return pairRanking


# In[260]:

def outputSeperate(seq):
    errLst = re.findall(r'\(.\)', seq)
    tmpStr = re.subn(r'\(.\)', '|||', seq)[0].split('|||')
    total_length = 0
    error_info = []
    for idx, cor_ch in zip(tmpStr[:-1], errLst):
        error_info.append((len(idx)+total_length, cor_ch[1]))
        total_length += len(idx)

    return ''.join(tmpStr), error_info


# In[271]:

def outputTestData(pairRanking, needRank, outputFile):
    wp = open(outputFile, 'w', encoding='utf8')
    for pair in pairRanking.keys():
        for seq in testData_pair[pair]:
            cleanSeq, error_info = outputSeperate(seq)
            error_infoStr = ', '.join(['{}, {}'.format(idx,ch) for idx, ch in error_info])
            wp.write('{}|||{}\n'.format(cleanSeq, error_infoStr))
    wp.close()


# In[189]:

### Global 
testData_pair = defaultdict(set)
testData_output = dict()
error_type = defaultdict(lambda :[0,0])
discard = []
UDN_benchmark_log = './UDN_benchmark/UDN_benchmark.log'


# In[278]:

def garbageDump(filename):
    with open(filename, 'w', encoding='utf8') as fp:
        fp.write('\n'.join(discard))


# In[280]:

def main(args):
    labelFile = args.input
    outputFile = args.output
    garbageFile = args.garbage
    
    if os.path.exists(UDN_benchmark_log):
        os.remove(UDN_benchmark_log)
    
    print('=== Reading {} ==='.format(labelFile))
    confused = loadData(labelFile)

    print('=== Conufsed label ===')
    confusedLabelClassify(confused)

    print('=== Mutual Case ===')
    mutualCase()

    print('=== Rank error type ===')
    pairRanking = rankPair()

    print('=== Output {} ==='.format(outputFile))
    outputTestData(pairRanking, [], outputFile)
    
    print('=== Dump discard to ==='.format(garbageFile))
    garbageDump(garbageFile)


# In[275]:

def processCommand():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o','--output', required=True)
    parser.add_argument('-i', '--input', default='./withError_label.txt')    
    parser.add_argument('--garbage', default='./UDN_benchmark/UDN_discard.txt')
    
    return parser.parse_args()


# In[ ]:

if __name__ == '__main__':
    args = processCommand()
    
    main(args)

