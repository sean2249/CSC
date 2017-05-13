
# coding: utf-8

# In[1]:

import pickle
import numpy as np
from sklearn import svm


# In[61]:

def extract(filename, threshold):
    with open(filename, 'rb') as fp:
        bigDict = pickle.load(fp)
    # create label 
    label = list()
    feature = list()

    # idx = 0 
    for error_ch, (cands_val) in bigDict.items():
    #     if idx>50: break
    #     idx += 1
        # two situation for error: (error-pair) or (higher-score)
        for cand, (score, log) in cands_val.items():
            # Feature 
            tmp = [i for i in log[:4]]
            tmp.extend(log[4])
            tmp.append(log[-1])

            feature.append(tmp)

            # Positive case (error)
            if len(log) > 6 or score > threshold:
                label.append(1)
            else:
                label.append(2)

    train_feature = np.asarray(feature, dtype='float')            
    tmp_label = np.asarray(label, dtype='int')
    pos = np.where(tmp_label == 1 )[0]
    neg = np.where(tmp_label == 2 )[0]
    train_label = {1:pos, 2:neg}        

    print('The number of sample = {}'.format(train_feature.shape))
    print('Positive case (candidate) = {}'.format(len(train_label[1])))
    print('Negative case (uncandidate) = {}'.format(len(train_label[2])))
    
    return (train_feature, train_label)


# In[317]:

def train(feature, label, train_cnt, test_cnt=[]):

    # If test_cnt not declare, use all the remain set as test set
    if not test_cnt:
        test_cnt = len(label[1]) - train_cnt
        
    # Picke feature/label to train & test set    
    np.random.shuffle(label[1])
    np.random.shuffle(label[2])

    train_idx = np.concatenate(
        (label[1][:train_cnt],
         label[2][:train_cnt]))
    train_label = np.concatenate(
        (np.full(train_cnt, 1, dtype=int), np.full(train_cnt, 2, dtype=int)))
    train_feature = feature[train_idx]

    test_idx = np.concatenate(
        (label[1][train_cnt:train_cnt+test_cnt], 
         label[2][train_cnt:train_cnt+test_cnt]))
    test_label = np.concatenate(
        (np.full(test_cnt, 1, dtype=int), np.full(test_cnt, 2, dtype=int)))
    test_feature = feature[test_idx]

    # Training 
    clf = svm.SVC(kernel='rbf')
    clf.fit(train_feature,train_label)
    
    # Testing 
    print(clf.score(test_feature, test_label))
    
    return clf


# In[62]:

if __name__ == '__main__':
    filename = 'confu.pkl'
    
    (feature, label) = extract(filename, 10)


# In[318]:

    train_set = [3,5,10,20,50,100,200,300,500,700,1000,1500,2000,3000,5000,7000,10000,15000,17000,20000]


# In[311]:

    for t in train_set:
        clf = train(feature, label, t)

