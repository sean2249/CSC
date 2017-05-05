import pickle


class NCM:
    def __init__(self, channel_filename):
        print('Loading channel model %s ...' %(channel_filename))
        with open(channel_filename, 'rb') as fp:
            self.table = pickle.load(fp, encoding='utf8')
        #  self.table = pickle.load(open(channel_filename,'rb'), encoding='utf8')
    def cand(self, cur_char, show=0):
        query_cands = []
        if cur_char in self.table:
            query_cands = self.table[cur_char].items()            
            
        if show==1:
            for cands in query_cands:
                print(cands)
        return query_cands
