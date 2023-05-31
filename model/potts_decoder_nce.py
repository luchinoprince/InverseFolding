## In this file we create the model for NCE which has an additional parameter, the normalising constant

import torch
from torch.nn import Linear, TransformerEncoderLayer, LayerNorm
from torchvision.ops import MLP
from torch.autograd import profiler 
import numpy as np

class PottsDecoder(torch.nn.Module):

    def __init__(self, q, n_layers, d_model, input_encoding_dim, param_embed_dim, n_heads, n_param_heads, dropout=0.0):

        super().__init__()
        self.q = q
        self.n_layers = n_layers
        self.d_model = d_model
        self.input_encoding_dim = input_encoding_dim
        self.param_embed_dim = param_embed_dim
        self.n_heads = n_heads
        self.n_param_heads = n_param_heads
        self.dropout = dropout

        print(self.input_encoding_dim)
        print(self.d_model)
        self.input_MLP = Linear(self.input_encoding_dim, self.d_model)
        #self.input_MLP = MLP(self.input_encoding_dim, hidden_channels=[self.d_model, self.d_model], bias=False, norm_layer=LayerNorm, dropout=0.0, inplace=False)
        #self_input_norm = LayerNorm()

        self.attention_layers = torch.nn.ModuleList([])
        self.relu = torch.nn.ReLU()
        for _ in range(n_layers):
            attention_layer = TransformerEncoderLayer(self.d_model, self.n_heads,
                                                      dropout=self.dropout, batch_first=True)
            self.attention_layers.append(attention_layer)

        ## Here we do a projection so that we can have the sum of the rank-one matrices as we want

        #self.P = MLP(self.d_model, hidden_channels=[self.d_model, self.d_model*self.n_param_heads], bias=False, norm_layer=LayerNorm, dropout=0.0, inplace=False)

        #self.P = torch.nn.Parameter(torch.randn(self.n_param_heads, self.d_model, self.d_model), requires_grad=True)
        
        self.P = Linear(self.d_model, self.n_param_heads*self.d_model, bias=False)   ## this uses a more sensible initialization
        
        #self.P.retain_grad()
        #self.P.register_hook(lambda x: print("backward called"))

        self.output_linear = Linear(self.d_model, self.q)

        self.field_linear = Linear(self.q, self.q)
        self.logZ = Linear(self.q, 1)

    
    def _get_params(self, param_embeddings, N, padding_mask):
        ## param_embeddings: (B, n_param_heads, N, q) 
        ## (B, n_param_heads, N, q) * (B, 1, N, 1) = (B, K, N, q)
        # set embeddings to zero where padding is present
        padding_mask_inv = (~padding_mask)
        param_embeddings = param_embeddings * padding_mask_inv.unsqueeze(1).unsqueeze(3) 

        ## Get logZ (B, K, N, q) ---> (B, K, N, 1) --> (B, N, 1) --> (B, 1) --> (B, )
        logZ = torch.sum(torch.sum(self.logZ(param_embeddings), dim=1), dim=1).squeeze(-1) ## I don't think here you need to rescale. 

        # get fields: (B, K, N, q) ---> (B, K, N, q) ---> (B, N, q)
        fields = torch.sum(self.field_linear(param_embeddings), dim=1) *  self.n_param_heads**(-1/2)

        # set fields to 0 depending on the padding
        fields = fields * padding_mask_inv.unsqueeze(2)

        ## To normalize later computations
        param_embeddings = param_embeddings * self.n_param_heads**(-1/4)

        # flatten fields
        #fields = fields.view(-1, N * self.q)

        # flatten to (B, n_param_heads, N*q)
        #param_embeddings = param_embeddings.flatten(start_dim=2, end_dim=3)


        return param_embeddings, fields, logZ


    
    def forward(self, encodings, padding_mask):

        B, N, _ = encodings.shape

        assert B == padding_mask.shape[0]
        assert N == padding_mask.shape[1]

        embeddings = self.input_MLP(encodings)
        for attention_layer in self.attention_layers:
            embeddings = attention_layer(embeddings, src_key_padding_mask=padding_mask)
            embeddings = self.relu(embeddings)

        ## self.P(embeddings): (B, N, d_model) ---> (B, N, d_model*n_param_heads)
        ## We the reshape to (B, N, d_model, n_param_heads)


        ### THIS OPERATION SEEMS TO BE SUPER EXPENSIVE COMPUTATIONALLY, DON'T KNOW WHY
        #param_embeddings = self.P.unsqueeze(0).unsqueeze(2) @ embeddings.unsqueeze(1).unsqueeze(4)
        #param_embeddings = param_embeddings.squeeze()

        param_embeddings = torch.transpose(self.P(embeddings).reshape(B, N, self.n_param_heads, self.d_model), 1, 2) #@ embeddings.unsqueeze(1).unsqueeze(4)
        # (1, n_param_heads, 1, d_model, d_model) x (B, 1, N, d_model, 1) -> (B, n_param_heads, N, d_model)
        param_embeddings = self.relu(param_embeddings)

        # (B, n_param_heads, N, d_model) --> (B, n_param_heads, N, q)
        param_embeddings = self.output_linear(param_embeddings)
        param_embeddings, fields, logZ = self._get_params(param_embeddings, N, padding_mask)

        return param_embeddings, fields, logZ







