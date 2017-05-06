class CANDSEXPAND:
    def __init__(self, ncmEx_filename):
        self.prefix = defaultdict(set)
        self.postfix = defaultdict(set)
        with open(ncmEx_filename, 'r', encoding='utf8') as fp:
            for line in fp:
                seq = line.strip()
                if len(seq)==2:
                    self.prefix[seq[1]].add(seq[0])
                    self.postfix[seq[0]].add(seq[1])
    def cand(self, pre_char, post_char, rank=0.05, show=0):
        # Union postfix & prefix
        tmpall = self.postfix[pre_char]|self.prefix[post_char]
        ncm_stats = namedtuple('ncm_prob', 'ch,prob')
        for item in tmpall:
            yield item      