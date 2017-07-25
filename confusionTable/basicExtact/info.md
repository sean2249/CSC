
```python
# Sound Unihan 
if step>=1 or step == -1:
    cands.update(sound_extract_same(ch_x))
    cands.update(sound_extract_similartConsonant(ch_x))
    cands.update(sound_extract_tone(ch_x))
    cands.update(sound_extract_finalConsonant(ch_x))
if SHOW == 1: print(len(cands))

# Shape Unihan 
if step >= 2 or step == -2:
    cands.update(shape_similar(ch_x))        
if SHOW == 1: print(len(cands))

# Sound SIGHAN
if step >= 3 or step == -3:
    try:
        for col in sound_SIGHAN.loc[ch_x][:-1]:
            if type(col)==str:
                cands.update(list(col))
    except KeyError:
        pass
if SHOW == 1: print(len(cands))

# Shape SIGHAN similar shape
if step >= 4 or step == -4:
    cands1 = shape_SIGHAN.get(ch_x, [])
    cands.update(cands1)
if SHOW == 1: print(len(cands))

# Shape SIGHAN same compoent 
if step >= 5 or step == -5:
    try:
        cands2 = sound_SIGHAN.loc[ch_x].同部首同筆畫數
        if type(cands2)==float:
            cands2 = []
    except KeyError:
        cands2 = []
    cands.update(cands2)
if SHOW == 1: print(len(cands))

# confusion pair
if step >= 6 or step == -6:
    cands.update(combine_confusion_chPairs[ch_x])
if SHOW == 1: print(len(cands))

```
