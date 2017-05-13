#!/bin/bash
# $1: UDN_dataroot
# $2: token_path 
# $3: vocabulary
if [ $# -eq 3 ];then
    echo "Language corpus extract"
    python3 dataSegment4LMtraining.py $1 $2.txt

    ngram-count -text $2.txt -write $2.cnt \
    -order 4 -vocab $3
    ngram-count -read $2.cnt -lm $2.lm \
    -order 4 -vocab $3 -unk \
    -gt1min 5 -gt2min 5 -gt3min 5 -gt4min 5

    echo 'Done.'
    echo 'Language model $2.lm'
elif [ $# -eq 2 ]; then 
    echo "Language corpus extract"
    python3 dataSegment4LMtraining.py $1 $2.txt

    ngram-count -text $2.txt -write $2.cnt \
    -order 4 
    ngram-count -read $2.cnt -lm $2.lm \
    -order 4  -unk \
    -gt1min 5 -gt2min 5 -gt3min 5 -gt4min 5

else
    echo "Usage $0 \$1-UDN \$2-token_path \$4-vocabulary(optional)"
    exit 0
fi 

echo 'Done.'
echo 'Language model $3.lm'




