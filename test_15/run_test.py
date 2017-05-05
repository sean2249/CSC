import os 
import sys

# System argument value 
#  sys_truth = sys.argv[1]
#  test_num = sys.argv[2]
#  rmchar = sys.argv[3]

# Remove char for sub task 1
#  if test_num==1 & rmchar==1:
    #  print

def rmchar(sys_truth):
    outputname = os.path.splitext(sys_truth)[0]+'_rmch.txt'

    with open(sys_truth, 'r', encoding='utf8') as fp, open(outputname, 'w') as wp:
        for line in fp:
            line = line.strip()
            sep = line.split(', ')
            #  print(sep[1::2])
            wp.write(sep[0]+', ')
            wp.write(', '.join(sep[1::2]))
            wp.write('\n')
    return outputname


def run_eval(sys_truth, test_num):
    task = {'1':'SIGHAN15_CSC_DryTruth.txt','2':'SIGHAN15_CSC_TestTruth.txt'}
    judge = 'sighan15csc.jar'

    detail_path = 'de_'+os.path.splitext(sys_truth)[0]+'.txt'
    cmd_eval = 'java -jar %s -s %s -i %s -t %s -o %s'\
            %(judge, test_num, sys_truth, task[test_num] , detail_path)
    #  print(cmd_eval)
    if os.path.exists(judge) & os.path.exists(task[test_num]):
        os.system(cmd_eval)
    print('Output file = '+detail_path)
        

if __name__ == "__main__":
    if len(sys.argv)>=3:
        sys_truth = sys.argv[1]
        test_num = sys.argv[2]
        
        if not os.path.exists(sys_truth):
            print('File:{} not exists.'.format(sys_truth))
            sys.exit(0)

        if len(sys.argv)==4:
            print('Doing remove character...')
            sys_truth = rmchar(sys_truth)
        run_eval(sys_truth, test_num)
    else:
        print('Usage: python3 **.py sys_truth_filename test_num (rm char)')
        print('test_num: Dry=1 Test=2')
        sys.exit(0)

            
        
