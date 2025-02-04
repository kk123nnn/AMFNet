import torch, copy
import torch.nn as nn
import torch.nn.functional as F

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])    
    

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward = 2048, dropout = 0.1, activation = "relu", layer_norm_eps = 1e-5):
        super(TransformerEncoderLayer, self).__init__()

        # d_model is emb_size
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)


    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)


    def forward(self, src, src_mask = None, src_key_padding_mask = None):

        # Implementation of Multihead attention
        # The three 'src' correspond to Q, K, V, with dimensions being seq * batch * embedding.
        # 'src2' has the same dimension as 'src'
        # 'attn' should be batch * seq * seq
        # if Q, K, V is input_size * batch * embedding, then attn should be batch * input * input
        src2, attn = self.self_attn(src, src, src, attn_mask = src_mask,
                              key_padding_mask = src_key_padding_mask)

        # Add & Norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed Forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        # Add & Norm
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn



class TransformerEncoder(nn.Module):

    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, device, norm = None):
        super(TransformerEncoder, self).__init__()
        self.device = device
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm


    def forward(self, src, mask = None, src_key_padding_mask = None):

        output = src
        attn_output = torch.zeros((src.shape[1], src.shape[0], src.shape[0]), device = self.device) # batch, seq_len, seq_len
        
        for mod in self.layers:
            output, attn = mod(output, src_mask = mask, src_key_padding_mask = src_key_padding_mask)
            attn_output += attn
            
        if self.norm is not None:
            output = self.norm(output)

        return output, attn_output
