class myVocab:
    spl = '<p>'
    pad = '<pad>'
    eos = '</s>'
    unk = '<unk>'
    p1 = '<p1>'
    p2 = '<p2>'

    def __init__(self, vocab_file):
        # TODO: add check for special tokens
        self.spec_tokens = [myVocab.spl, myVocab.pad, myVocab.eos, myVocab.unk]
        with open(vocab_file, 'r', encoding='utf8') as fr:
            vocab = [line.strip('\n').split()[0] for line in fr.readlines()]
        vocab = self.spec_tokens + vocab
        self.spec_tokens = [myVocab.spl, myVocab.pad, myVocab.eos, myVocab.unk, myVocab.p1, myVocab.p2]
        self.token2id = {t: i for i, t in enumerate(vocab)}
        self.id2token = {i: t for i, t in enumerate(vocab)}

    def __len__(self):
        return len(self.token2id)

    @property
    def n_special_tokens(self):
        return len(self.spec_tokens)

    @property
    def special_tokens_ids(self):
        return [self.token2id[t] for t in self.spec_tokens]

    @property
    def pad_id(self):
        return self.token2id[myVocab.pad]

    @property
    def spl_id(self):
        return self.token2id[myVocab.spl]

    @property
    def p1_id(self):
        return self.token2id[myVocab.p1]

    @property
    def p2_id(self):
        return self.token2id[myVocab.p2]

    @property
    def bos_id(self):
        return self.token2id[myVocab.eos]

    @property
    def eos_id(self):
        return self.token2id[myVocab.eos]

    def string2ids(self, string):
        tokens = string.split()
        ids = [self.token2id[t] for t in tokens if t in self.token2id]
        return ids

    def ids2string(self, ids):
        tokens = [self.id2token[id] for id in ids]
        return ''.join(tokens)

