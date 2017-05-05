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


def eval(sys_truth, test_num):
    task = {'1':'FinalTest_SubTask1_Truth.txt','2':'FinalTest_SubTask2_Truth.txt'}
    judge = 'sighan7csc.jar'

    detail_path = os.path.splitext(sys_truth)[0]+'_detail.txt'
    cmd_eval = 'java -jar %s -s %s -i %s -t %s -o %s'\
            %(judge, test_num, sys_truth, task[test_num] , detail_path)
    print(cmd_eval)
    if os.path.exists(judge) & os.path.exists(task[test_num]):
        os.system(cmd_eval)
        

if __name__ == "__main__":
    if len(sys.argv)>=3:
        sys_truth = sys.argv[1]
        test_num = sys.argv[2]
        if len(sys.argv)==4:
            print('Doing remove character...')
            sys_truth = rmchar(sys_truth)
        eval(sys_truth, test_num)

            
        
