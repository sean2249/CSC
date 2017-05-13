### Chinese Spelling Check
1. charSpellingCheck: main program, run SIGHAN_2015
2. model/ncm,lm

### Extract UDN content
1. extractUDN.ipynb: extract the typos and sentence on UDN
2. UDN_errorFrequency: find the relation on typos based on the output from previous program

### LM training
0. runLM_training.sh: run "dataSegment4LMtraining.py" and "SRILM" 
1. dataSegment4LMtraining.py: extract the content on UDN on char/word based level 

### Confusion Model Create
1. confusionTable_extractFeature: extract lots of feature from different files 
2. confusionTable_organize: create poor confusion model 
3. confusionTable_ML: use machine learning to construct model 

