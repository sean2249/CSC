
# coding: utf-8

# In[2]:

import os 
import csv 
import argparse
import re
from datetime import datetime


# In[3]:

def run_eval(filelist, test_num, path):
    task = {'1': 'SIGHAN15_CSC_DryTruth.txt',
            '2': 'SIGHAN15_CSC_TestTruth.txt'}
    judge_file = 'sighan15csc.jar'

    for sys_file, sys_path in filelist.items():
#     (de, t) = run_eval(i, 2, path)
    
#     (sys_file, sys_path) = sys_truth
#         sys_token = sys_file[3:-4]

    #     sys_filename = os.path.splitext(sys_truth[0])[0]
    #     sys_token = sys_filename[sys_filename.find('_')+1:]

        detail_path = '{}{}'.format(path, re.sub(r're', 'de', sys_file))

    #     detail_path = 'de_' + sys_token + '.txt'
    #     cmd_eval = 'java -jar %s -s %s -i %s -t %s -o %s'\
    #         % (judge, test_num, sys_truth, task[test_num], detail_path)


        cmd_eval = 'java -jar {} -s {} -i {} -t {} -o {}'                .format(judge_file, str(test_num), sys_path, task[str(test_num)], detail_path)



    #     print(cmd_eval)
        if os.path.exists(judge_file) & os.path.exists(task[str(test_num)]):
            os.system(cmd_eval)

        print('Output file = {}'.format(detail_path))

#     return (detail_path, sys_token)


# In[4]:

def getVal(filename):
    output = list()
    with open(filename, 'r', encoding='utf8') as fp:
        for idx,line in enumerate(fp,1):
            # False Positive
            if idx == 7:
                output.append(line.split()[4])
            elif idx >= 10:
                tmp = line.split()
                if len(tmp)==4:
                    output.append(tmp[2])
            elif idx > 22:
                break

    return output


# In[5]:

def getResultList(path):
    filelist = dict()
    for dirpath, dirnames, filenames in os.walk(path):        
        for file in filenames:
            if re.search(r'^re', file):
                filelist[file] = '{}{}'.format(dirpath, file) 
    return filelist


# In[12]:

def getDetailList(path):
    filelist = dict()
    for dirpath, _, filenames in os.walk(path):
        for file in filenames:
            if re.search(r'^de', file):
                filelist[file] = '{}{}'.format(dirpath,file)
    
    filelist_sort = sorted(filelist.items(), key=lambda x:x[0])
    
    log_files = []
    for (file, filepath) in filelist_sort:                
        output = [file[3:-4]]
        with open(filepath, 'r', encoding='utf8') as fp:
            for idx,line in enumerate(fp,1):
                # False Positive
                if idx == 7:
                    output.append(line.split()[4])
                elif idx >= 10:
                    tmp = line.split()
                    if len(tmp)==4:
                        output.append(tmp[2])
                elif idx > 22:
                    break
        log_files.append(output)
    
    t = datetime.now()
    log_token = '{:02}{:02}_{:02}{:02}{:02}'.format(t.month,t.day,t.hour,t.minute,t.second)
    log_filename = '{}/log_{}.csv'.format(path, log_token)
    with open(log_filename, 'w', encoding='utf8', newline='') as fp:
        print('Log file = {}'.format(log_filename))            
        writer = csv.writer(fp) 
        writer.writerows(log_files)
    
#     print(filelist)
#     print(log_files)


# In[61]:

def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--path', required=True)
    parser.add_argument('-n','--num', required=True)
    
    return parser.parse_args()


# In[ ]:

def main(args):    
    filelist = getResultList(args.path)
    run_eval(filelist,args.num,args.path)
    getDetailList(args.path)


# In[ ]:

if __name__ == '__main__':
    args = process_command()
    
    main(args)
    
    

