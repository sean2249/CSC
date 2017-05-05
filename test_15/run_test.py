import os
import sys
from datetime import datetime


# System argument value
#  sys_truth = sys.argv[1]
#  test_num = sys.argv[2]
#  rmchar = sys.argv[3]

# Remove char for sub task 1
#  if test_num==1 & rmchar==1:
#  print

def rmchar(sys_truth):
    outputname = os.path.splitext(sys_truth)[0] + '_rmch.txt'

    with open(sys_truth, 'r', encoding='utf8') as fp, open(outputname, 'w') as wp:
        for line in fp:
            line = line.strip()
            sep = line.split(', ')
            #  print(sep[1::2])
            wp.write(sep[0] + ', ')
            wp.write(', '.join(sep[1::2]))
            wp.write('\n')
    return outputname


def run_eval(sys_truth, test_num):
    task = {'1': 'SIGHAN15_CSC_DryTruth.txt',
            '2': 'SIGHAN15_CSC_TestTruth.txt'}
    judge = 'sighan15csc.jar'

    sys_filename = os.path.splitext(sys_truth)[0]
    sys_token = sys_filename[sys_filename.find('_')+1:]
    detail_path = 'de_' + sys_token + '.txt'
    cmd_eval = 'java -jar %s -s %s -i %s -t %s -o %s'\
        % (judge, test_num, sys_truth, task[test_num], detail_path)
    # print(cmd_eval)
    if os.path.exists(judge) & os.path.exists(task[test_num]):
        os.system(cmd_eval)
    print('Output file = ' + detail_path)

    return (detail_path, sys_token)

def getVal(filename, token):
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
    # with open('log_{}.txt'.format(token), 'w', encoding='utf8') as fp:
    #     fp.write('\t'.join(output))

    # print('Logfile = log_{}.txt'.format(token))
    # return 'log_{}.txt'.format(token)
            

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        TEST_NUM = sys.argv[1]
        
        SYS_TRUTHS = [filename for filename in sys.argv[2:]]

        log_files = list()
        for sys_file in SYS_TRUTHS:
            if not os.path.exists(sys_file):
                print('File:{} not exists.'.format(sys_file))
                sys.exit(0)

            (detail_filename, token) = run_eval(sys_file, TEST_NUM)
            log_files.append(getVal(detail_filename, token))
        
        
        
        t = datetime.now()
        log_token = '{:02}{:02}_{:02}{:02}{:02}'.format(t.month,t.day,t.hour,t.minute,t.second)
        with open('log_{}.txt'.format(log_token), 'w', encoding='utf8') as fp:
            print('Log file = log_{}.txt'.format(log_token))            
            for batch in log_files:
                fp.write('\t'.join(batch))
                fp.write('\n')
            






    else:
        print('Usage: python3 **.py sys_truth_filename test_num ')
        print('test_num: Dry=1 Test=2')
        sys.exit(0)
