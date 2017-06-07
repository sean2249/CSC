
# coding: utf-8

# In[2]:

import argparse
import pickle


# In[3]:

seqTruth = './test_15/SIGHAN15_CSC_TestInput.txt'
groundTruth = './test_15/SIGHAN15_CSC_TestTruth.txt'


# In[5]:

def lineSeperate(line):
    _log = line.strip().split(', ')
    if len(_log) > 2:
        _id, _result = _log[0], dict((a,b) for a,b in zip(_log[1:-1:2], _log[2::2]))
    else:
        _id, _result = _log[0], dict()
    
    return _id, _result


# In[6]:

def seqSeperate(line):
    _log = line.strip()
    
    idx1 = _log.find('=')+1
    idx2 = _log.find(')')
    
    _id = _log[idx1:idx2]
    _seq = _log[idx2+2:]
    
    return _id, _seq


# In[17]:

def process(seq_log, gt_log, sys_log, ncm):
    global corrcnt, encnt, elcnt, mcnt
    
    label = dict()            
    case = [
        'EN','EL','M']
    
    
    keys = set(gt_log.keys()).union(set(sys_log.keys()))
    
    for key in keys:
        error_ch = seq_log[int(key)-1]
        sys_ch = sys_log.get(key, 'x')
        correct_ch = gt_log.get(key, 'x')
        
        # Correct
        if sys_ch == correct_ch:
            label[key] = '({})'.format(correct_ch) 
            corrcnt += 1
        
        # Have error
        elif correct_ch != 'x':
            flag = bool(ncm.get(error_ch,{}).get(correct_ch))
            if flag:
                label[key] = '({}|{}|{})'.format(sys_ch, correct_ch, case[0])
                encnt += 1
            else:
                label[key] = '({}|{}|{})'.format(sys_ch, correct_ch, case[1])
                elcnt += 1
                
        # Dont' have error
        else:
            label[key] = '({}|{}|{})'.format(sys_ch, correct_ch, case[2])
            mcnt += 1
            
    label = sorted(label.items(), key=lambda x:int(x[0]))
#     print(label)
    
    output = list(seq_log)
    extend = 0
    for idx, detail in label:
        output.insert(int(idx)+extend, detail)
        extend += 1
    
    return ''.join(output)


# In[22]:

def main(args):
    confuFile = args.confusion
    sysTruth = args.systruth
    outputFile = args.output
    
    
    with open(confuFile, 'rb') as fp:
        ncm = pickle.load(fp)

    output = []
    with open(seqTruth, 'r', encoding='utf8') as seqp,        open(groundTruth, 'r', encoding='utf8') as gp,        open(sysTruth, 'r', encoding='utf8') as sysp:
            
            for seq_line, gt_line, sys_line in zip(seqp, gp, sysp):                
                seq_id, seq_log = seqSeperate(seq_line)                        
                gt_id, gt_log = lineSeperate(gt_line)
                sys_id, sys_log = lineSeperate(sys_line)                

                if seq_id == gt_id and gt_id == sys_id:
                    output.append(process(seq_log, gt_log, sys_log, ncm))

    with open(outputFile, 'w', encoding='utf8') as fp:
        fp.write('\n'.join(output))


# In[ ]:

def process_command():
    parse = argparse.ArgumentParser()
    parse.add_argument('-s', '--systruth', required=True)
    parse.add_argument('-c', '--confusion', required=True)
    parse.add_argument('-o','--output', required=True)
    
    return parse.parse_args()


# In[ ]:

corrcnt, encnt, elcnt, mcnt = 0, 0, 0, 0


# In[ ]:

if __name__ == '__main__':    
    args = process_command()
    
    
    
    main(args)
    
    print('Correct:{}\tErrorNCM:{}\tErrorLack:{}\tMiss:{}'.format(corrcnt, encnt, elcnt, mcnt))

