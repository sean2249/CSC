
# coding: utf-8

# In[1]:

import os


# In[2]:

filename = './withError_format.txt'
outputname = './withError_label.txt'


# In[3]:

with open(filename, 'r', encoding='utf8') as fp:
    seqs = [i.strip() for i in fp.readlines()]


# In[5]:

SPELL = ';'
EDITOR = '\''
REDO = 'r'

SPELL_TAG = 'S'
EDITOR_TAG = 'E'


# In[16]:

if os.path.exists(outputname):
    with open(outputname, 'r', encoding='utf8') as fp:
        end_idx = len(fp.readlines())-1
else:
    end_idx = -1
print('Start from {}'.format(end_idx))


# In[17]:

a = 0
print('{} == spelling error'.format(SPELL))
print('{} == editor error'.format(EDITOR))
for idx, s in enumerate(seqs):    
    if idx <= end_idx:
        continue
    
    while a!=SPELL and a!=EDITOR:    
        a = input('{}{}\t'.format(idx, s))
    
    if a == SPELL:
        sp = '{}{}\n'.format(SPELL_TAG, s)
        
    elif a == EDITOR:
        sp = '{}{}\n'.format(EDITOR_TAG, s)

    else:
        sys.exit(0)
        
    with open(outputname, 'a', encoding='utf8') as wp:
        wp.write(sp)
    
    a = 0
        
    
    
    

