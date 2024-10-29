import torch
from torchtext.vocab import GloVe, vocab

class GloveTokenizer:
    def __init__(self, max_length=140):
        self.max_length = max_length
        self.vocabulary = self._build_vocab()
        
    def _build_vocab(self):
        pretrained_embeddings = GloVe(name='twitter.27B', dim=25)
        glove_vocab = vocab(pretrained_embeddings.stoi, min_freq=0)
        glove_vocab.insert_token('<unk>', 0)
        glove_vocab.insert_token('<pad>', 0)
        glove_vocab.set_default_index(1)

        return glove_vocab

    def tokenize(self, tweets):
        tokenized_tweets = []
        
        for tweet in tweets:
            tokens = self.vocabulary(tweet.strip().split())
            if len(tokens) < self.max_length:
                tokens += [0] * (self.max_length - len(tokens))
            else:
                tokens = tokens[:self.max_length]
            tokenized_tweets.append(tokens)

        return torch.tensor(tokenized_tweets)