import torch
from torch import nn
from torchtext.vocab import GloVe

class LSTMClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding.from_pretrained(self._get_embeddings(), freeze=True)
        if self.config['num_layers'] == 1:
            self.lstm = nn.LSTM(self.config['embedding_dim'], self.config['hidden_dim'], num_layers=self.config['num_layers'], 
                                batch_first=True, bidirectional=True)
            self.fc = nn.Linear(self.config['hidden_dim'], 1)
        else:
            self.lstm = nn.LSTM(self.config['embedding_dim'], self.config['hidden_dim'], num_layers=self.config['num_layers'], 
                                batch_first=True, dropout = self.config['dropout'], bidirectional=True)
            self.fc = nn.Linear(self.config['hidden_dim'] * 2, 1)
        self.dropout = nn.Dropout(self.config['dropout'])

    def _get_embeddings(self):
        pretrained_embeddings = GloVe(name='twitter.27B', dim=self.config['embedding_dim'])
        pad_vector = torch.zeros(1, self.config['embedding_dim'])
        unk_vector = torch.mean(pretrained_embeddings.vectors, dim=0, keepdim=True)
        embeddings = torch.cat((pad_vector, unk_vector, pretrained_embeddings.vectors))

        return embeddings
        
    def forward(self, x):
        x = self.embedding(x)
        output, (hidden_states, cell_states) = self.lstm(x)
        if self.config['num_layers'] == 1:
            hidden = hidden_states[-1,:,:]
        else:
            hidden = torch.cat((hidden_states[-2,:,:], hidden_states[-1,:,:]), dim = 1)
        outputs = self.dropout(hidden)
        outputs = self.fc(outputs)

        return outputs