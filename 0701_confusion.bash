#!/bin/bash

i=1
kiwi[1]=3
echo ${kiwi[$i]}
confu[1]="./confusionTable/0627_select/confu_preFnbF_preFnbF_1000_5_0225_9995.pkl"
confu[2]="./confusionTable/0627_select/confu_preFnbF_preFnbF_1000_5_0225_99995.pkl"
confu[3]="./confusionTable/0627_select/confu_preFnbF_preFnbF_1000_5_0225_999995.pkl"
confu[4]="./confusionTable/0627_select/confu_preFnbF_preFnbF_1000_5_0225_9999995.pkl"
confu[5]="./confusionTable/0627_select/confu_preFnbT_preFnbT_1000_5_0225_9995.pkl"
confu[6]="./confusionTable/0627_select/confu_preFnbT_preFnbT_1000_5_0225_99995.pkl"
confu[7]="./confusionTable/0627_select/confu_preFnbT_preFnbT_1000_5_0225_999995.pkl"
confu[8]="./confusionTable/0627_select/confu_preFnbT_preFnbT_1000_5_0225_9999995.pkl"


for n in $(seq 2 1 3);do 
    for w in $(seq 5 1 8);do 
        for idx in $(seq 1 1 8);do 

            CONFUSION=${confu[$idx]}
            WEIGHT=$(echo $w | awk '{printf("%.1f",$1/10)}')
            CHPORT=$(echo $w | awk '{printf("%d", 5450+$1-5)}')
            WORDPORT=$(echo $w | awk '{printf("%d",5400+$1-5)}') 
            FILE1=$(echo $n $w $idx | awk '{printf("vitAll_%s_%s_%s",$3,$1,$2)}')
            FILE2=$(echo $n $w $idx | awk '{printf("vitEle_%s_%s_%s",$3,$1,$2)}')
            
            python charSpellingCheck.py --token $FILE1 --ngnum $n --weight $WEIGHT --chport $CHPORT --wordport $WORDPORT --method v --methodsmall all --ncm $CONFUSION &
            python charSpellingCheck.py --token $FILE2 --ngnum $n --weight $WEIGHT --chport $CHPORT --method v --methodsmall element --ncm $CONFUSION &
        done
    done
done
