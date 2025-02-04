import torch
import torch.nn as nn
import math
import transformer



class PositionalEncoding(nn.Module):

    def __init__(self, seq_len, d_model, dropout = 0.1):
        super(PositionalEncoding, self).__init__()
        max_len = max(5000, seq_len)
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[: , 0 : -1]
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    # Input: seq_len x batch_size x dim
    # Output: seq_len x batch_size x dim
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
    
    
class Permute(torch.nn.Module):
    def forward(self, x):
        return x.permute(1, 0)
    
    

class MultitaskTransformerModel(nn.Module):

    def __init__(self, task_type, device, nclasses, seq_len, batch, input_size, emb_size, nhead, nhid, nhid_tar, nhid_task, nlayers, dropout = 0.1):
        super(MultitaskTransformerModel, self).__init__()
        
        # set up a learnable parameters to determine the masking threhold
        self.masking_threshold = nn.Parameter(torch.tensor(0.15))

        # for seq_len
        self.trunk_net1 = nn.Sequential(
            nn.Linear(input_size, emb_size),
            nn.BatchNorm1d(batch),
            PositionalEncoding(seq_len, emb_size, dropout),
            nn.BatchNorm1d(batch)
        )

        # for input_size
        self.trunk_net2 = nn.Sequential(
            nn.Linear(seq_len, emb_size),
            nn.BatchNorm1d(batch),
            PositionalEncoding(input_size, emb_size, dropout),
            nn.BatchNorm1d(batch)
        )
        
        encoder_layers = transformer.TransformerEncoderLayer(emb_size, nhead, nhid, dropout)
        self.transformer_encoder = transformer.TransformerEncoder(encoder_layers, nlayers, device)
        
        self.batch_norm = nn.BatchNorm1d(batch)
        
        # Task-aware Reconstruction Layers
        self.tar_net = nn.Sequential(
            nn.Linear(emb_size, nhid_tar),
            nn.BatchNorm1d(batch),
            nn.Linear(nhid_tar, nhid_tar),
            nn.BatchNorm1d(batch),
            nn.Linear(nhid_tar, input_size),
        )

        if task_type == 'classification':
            # Classification Layers
            self.class_net = nn.Sequential(
                nn.Linear(emb_size, nhid_task),
                nn.ReLU(),
                Permute(),
                nn.BatchNorm1d(batch),
                Permute(),
                nn.Dropout(p = 0.3),
                nn.Linear(nhid_task, nhid_task),
                nn.ReLU(),
                Permute(),
                nn.BatchNorm1d(batch),
                Permute(),
                nn.Dropout(p = 0.3),
                nn.Linear(nhid_task, nclasses)
            )
        else:
            # Regression Layers
            self.reg_net = nn.Sequential(
                nn.Linear(emb_size, nhid_task),
                nn.ReLU(),
                Permute(),
                nn.BatchNorm1d(batch),
                Permute(),
                nn.Linear(nhid_task, nhid_task),
                nn.ReLU(),
                Permute(),
                nn.BatchNorm1d(batch),
                Permute(),
                nn.Linear(nhid_task, 1),
            )
            

        
    def forward(self, x, task_type):
        
        # x is batch * seq_len * input_size
        # After permute it should be seq_len * batch * input_size
        # After trunk_net, x1 should be seq_len * batch * embeding
        x1 = self.trunk_net1(x.permute(1, 0, 2))

        # x is batch * seq_len * input_size
        # After permute it should be input_size * batch * seq_len
        # After trunk_net, x2 should be input_size * batch * embeding
        x2 = self.trunk_net2(x.permute(2, 0, 1))
        
        # Transformer Encoder and calculate attention
        # x1 : seq_len * batch * embeding
        # attn1: batch * seq_len * seq_len
        x1, attn1 = self.transformer_encoder(x1)

        # attn2: batch * input_size * input_size
        attn2 = self.transformer_encoder(x2)[1]

        # sum up
        # time_weights_sum: batch_size * seq_len
        time_weights_sum = torch.sum(attn1, axis = 1) - torch.diagonal(attn1, offset = 0, dim1 = 1, dim2 = 2)

        # feature_weights_sum: batch_size * input_size
        feature_weights_sum = torch.sum(attn2, axis = 1) - torch.diagonal(attn2, offset = 0, dim1 = 1, dim2 = 2)

        # attn: batch_size * seq_size * num_fea
        attn = time_weights_sum.unsqueeze(2) * feature_weights_sum.unsqueeze(1)

        x1 = self.batch_norm(x1)

        if task_type == 'reconstruction':

            # tar_net(x): seq * batch * input
            # output: batch * seq * input
            output = self.tar_net(x1).permute(1, 0, 2)

        elif task_type == 'classification':

            output = self.class_net(x1[-1])

        elif task_type == 'regression':

            output = self.reg_net(x1[-1])

        return output, attn

