# 批量翻译
sentence_pairs = [
    ['je pars en vacances pour quelques jours .', 'i m taking a couple of days off .'],
    ['je ne me panique pas .', 'i m not panicking .'],
    ['je recherche un assistant .', 'i am looking for an assistant .'],
    ['je suis loin de chez moi .', 'i m a long way from home .'],
    ['vous etes en retard .', 'you re very late .'],
    ['j ai soif .', 'i am thirsty .'],
    ['je suis fou de vous .', 'i m crazy about you .'],
    ['vous etes vilain .', 'you are naughty .'],
    ['il est vieux et laid .', 'he s old and ugly .'],
    ['je suis terrifiee .', 'i m terrified .'],
]

import numpy as np
test_data_list = [
    " ".join([str(i) for i in list(np.random.randint(10,size=10))])
    for _ in range(20)
]
sentence_pairs = [
    [i, i] for i in test_data_list
]

all_words = []
for x, y in sentence_pairs:
    all_words.extend(x.split())
    all_words.extend(y.split())
all_words = sorted(list(set(all_words)))
vocab2id = {word: i for i, word in enumerate(all_words)}
id2vocab = {i: word for word, i in vocab2id.items()}
print(vocab2id)

class Tokenizer():
    def __init__(self, vocab2id) -> None:
        self.vocab_size = len(vocab2id) + 1
        self.padding_token_id = len(vocab2id)
        self.vocab2id = vocab2id
        self.id2vocab = {i: word for word, i in self.vocab2id.items()}

    def encode(self, sentences):
        ids = [self.vocab2id[w] for w in sentences.split()]
        return ids
    
    def decode(self, ids):
        return " ".join([self.id2vocab[i] for i in ids])

if __name__ == "__main__":
    tok = Tokenizer(vocab2id=vocab2id)
    s = sentence_pairs[0][0]
    print(s)
    ids = tok.encode(s)
    print(ids)
    print(tok.decode(ids))