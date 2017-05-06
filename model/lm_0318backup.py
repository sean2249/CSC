from collections import namedtuple, defaultdict
import math


class LM:
    def __init__(self, lm_filename):
        print('%s loading...' %(lm_filename))
        ngram_stats = namedtuple('lm_prob', 'p,bp')
        self.lm = {}
        with open(lm_filename,'r', encoding='utf8') as fp:
            for line in fp:
                seq = line.strip().split('\t')
                if len(seq)<2: continue
                (word, prob, backprob) = (''.join(seq[1].split()), 
                                          float(seq[0]), 
                                          float(seq[2]) if len(seq)==3 else 0.0)
                self.lm[word] = ngram_stats(prob, backprob)
    def score_batch(self, seq, show=0):
        ngram_stats = namedtuple('lm_prob', 'p,bp')
        failed_stat = ngram_stats(0.0,0.0)
        if show==1: 
            print('_____')
            print('First',seq)
        score_batch = 0.0        
        if seq in self.lm:
            score_batch += self.lm.get(seq, 0.0).p
        elif len(seq)==1:
            # Single character OOV 
            if show==1: print('OOV problem ', str(seq), ' prob=-99')
            score_batch += -99
        else:            
            if show==1: print('nofound:',seq[1:])
            fp = self.score_batch(seq[1:],show)
            bp = self.lm.get(seq[:-1], failed_stat).bp
            if bp==0:
                # Makesure back-off contain the character
                bp = self.score_batch(seq[:-1],show)
#             bp = self.lm.get(seq[:-1], self.score_batch(seq[:-1])).bp
            if show==1: 
                print('')
                print('fp+bp= ',fp,"+",bp)            
            score_batch = fp + bp
        if show==1: print('seq ',seq,'\nscore_batch',score_batch,'\n_____')        

        return score_batch
    
    def scoring(self, sentence, ngram=3,show=0):        
        seq = str(sentence)        
        score = 0.0
        for cur_idx in range(len(seq)):
            if show==1: print('========')
            idx_initial = cur_idx-ngram+1 if cur_idx-ngram>=0 else 0 
            idx_final = cur_idx+1
#             score += self.score_batch(seq[0:cur_idx+1],show)
            score += self.score_batch(seq[idx_initial:idx_final], show)            
            if show==1: print('tmp: ',score)
        if show==1: print(sentence,': ',score)
        return score