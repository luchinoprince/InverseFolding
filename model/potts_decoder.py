import torch
from torch.nn import Linear, TransformerEncoderLayer, LayerNorm
from torchvision.ops import MLP
from torch.autograd import profiler 
import numpy as np

from torch.nn.functional import softmax
from torch.distributions import Categorical

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

        #print(self.input_encoding_dim)
        #print(self.d_model)
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

    def _get_params(self, param_embeddings, N, padding_mask):
        
        padding_mask_inv = (~padding_mask)

        # set embeddings to zero where padding is present
        param_embeddings = param_embeddings * padding_mask_inv.unsqueeze(1).unsqueeze(3)

        # get fields ---> here I sum over K!
        fields = torch.sum(self.field_linear(param_embeddings), dim=1)

        # set fields to 0 depending on the padding
        fields = fields * padding_mask_inv.unsqueeze(2)

        # flatten fields
        fields = fields.view(-1, N * self.q)

        # flatten to (B, n_param_heads, N*q)
        param_embeddings = param_embeddings.flatten(start_dim=2, end_dim=3)

        # outer to (B, N*q, N*q)
        couplings = torch.einsum('bpi, bpj -> bij', (param_embeddings, param_embeddings))

        # create mask for couplings
        t = torch.ones(self.q, self.q)
        mask_couplings = (1 - torch.block_diag(*([t] * N))).to(couplings.device)
        mask_couplings.requires_grad = False

        couplings = couplings * mask_couplings

        return couplings/np.sqrt(self.n_param_heads), fields/np.sqrt(self.n_param_heads)
    
    def forward(self, encodings, padding_mask):

        B, N, _ = encodings.shape

        assert B == padding_mask.shape[0]
        assert N == padding_mask.shape[1]
        
        #with profiler.record_function("Embeddings"):
        embeddings = self.input_MLP(encodings)
        #with profiler.record_function("Attention Layers"):
        for attention_layer in self.attention_layers:
            embeddings = attention_layer(embeddings, src_key_padding_mask=padding_mask)
            embeddings = self.relu(embeddings)

        # (1, n_param_heads, 1, d_model, d_model) x (B, 1, N, d_model, 1) -> (B, n_param_heads, N, d_model)
        #with profiler.record_function("Projection pass"):
        ##OLD VERSION
        #param_embeddings = self.P.unsqueeze(0).unsqueeze(2) @ embeddings.unsqueeze(1).unsqueeze(4)
        #param_embeddings = torch.transpose(self.P.reshape(self.d_model, self.d_model, self.n_param_heads), 0, 2).unsqueeze(0).unsqueeze(2) @ embeddings.unsqueeze(1).unsqueeze(4)
        #param_embeddings = param_embeddings.squeeze()
        #param_embeddings.register_hook(lambda x: print("param embeddings backward called"))
        #breakpoint()

        param_embeddings = torch.transpose(self.P(embeddings).reshape(B, N, self.n_param_heads, self.d_model), 1, 2)
        #embeddings.register_hook(lambda x: print("embeddings backward called"))

        # apply relu
        param_embeddings = self.relu(param_embeddings)

        # (B, n_param_heads, N, q)
        param_embeddings = self.output_linear(param_embeddings)
        #with profiler.record_function("Get params"):
        couplings, fields = self._get_params(param_embeddings, N, padding_mask)

        return couplings, fields
    
    def _get_params_new(self, param_embeddings, N, padding_mask):
        
        padding_mask_inv = (~padding_mask)

        # set embeddings to zero where padding is present
        param_embeddings = param_embeddings * padding_mask_inv.unsqueeze(1).unsqueeze(3) 

        # get fields
        fields = torch.sum(self.field_linear(param_embeddings), dim=1) *  self.n_param_heads**(-1/2)

        # set fields to 0 depending on the padding
        fields = fields * padding_mask_inv.unsqueeze(2)

        ## To normalize later computations
        param_embeddings = param_embeddings * self.n_param_heads**(-1/4)

        # flatten fields
        #fields = fields.view(-1, N * self.q)

        # flatten to (B, n_param_heads, N*q)
        #param_embeddings = param_embeddings.flatten(start_dim=2, end_dim=3)


        return param_embeddings, fields


    
    def forward_new(self, encodings, padding_mask):

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

        # (B, n_param_heads, N, q)
        param_embeddings = self.output_linear(param_embeddings)
        param_embeddings, fields = self._get_params_new(param_embeddings, N, padding_mask)

        return param_embeddings, fields

    def _get_params_indep(self, param_embeddings, N, padding_mask):
        """ This is the function that forwards fot the model without couplings"""
        padding_mask_inv = (~padding_mask)

        # set embeddings to zero where padding is present
        param_embeddings = param_embeddings * padding_mask_inv.unsqueeze(1).unsqueeze(3) 

        # get fields
        fields = torch.sum(self.field_linear(param_embeddings), dim=1) *  self.n_param_heads**(-1/2)

        # set fields to 0 depending on the padding
        fields = fields * padding_mask_inv.unsqueeze(2)

        return fields


    
    def forward_indep(self, encodings, padding_mask):
        """ This is the forward function for the model that does not have the the Couplings"""
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

        # (B, n_param_heads, N, q)
        param_embeddings = self.output_linear(param_embeddings)
        fields = self._get_params_indep(param_embeddings, N, padding_mask)

        return fields
    
    def _get_params_ardca(self, param_embeddings, N, padding_mask):
        
        padding_mask_inv = (~padding_mask)

        # set embeddings to zero where padding is present
        param_embeddings = param_embeddings * padding_mask_inv.unsqueeze(1).unsqueeze(3)

        # get fields ---> here I sum over K!
        fields = torch.sum(self.field_linear(param_embeddings), dim=1)

        # set fields to 0 depending on the padding
        fields = fields * padding_mask_inv.unsqueeze(2)

        # flatten fields
        fields = fields.view(-1, N * self.q)

        # flatten to (B, n_param_heads, N*q)
        param_embeddings = param_embeddings.flatten(start_dim=2, end_dim=3)

        # outer to (B, N*q, N*q)
        couplings = torch.einsum('bpi, bpj -> bij', (param_embeddings, param_embeddings))

        # create mask for couplings
        t = torch.ones(self.q, self.q)
        mask_couplings = (1 - torch.block_diag(*([t] * N))).to(couplings.device)
        mask_couplings.requires_grad = False

        couplings = couplings * mask_couplings

        #### We keen only lower triangular since we want to do arDCA
        couplings = torch.tril(couplings)

        return couplings/np.sqrt(self.n_param_heads), fields/np.sqrt(self.n_param_heads)
    
    def forward_ardca(self, encodings, padding_mask):

        B, N, _ = encodings.shape

        assert B == padding_mask.shape[0]
        assert N == padding_mask.shape[1]
        
        #with profiler.record_function("Embeddings"):
        embeddings = self.input_MLP(encodings)
        #with profiler.record_function("Attention Layers"):
        for attention_layer in self.attention_layers:
            embeddings = attention_layer(embeddings, src_key_padding_mask=padding_mask)
            embeddings = self.relu(embeddings)

        param_embeddings = torch.transpose(self.P(embeddings).reshape(B, N, self.n_param_heads, self.d_model), 1, 2)
        #embeddings.register_hook(lambda x: print("embeddings backward called"))

        # apply relu
        param_embeddings = self.relu(param_embeddings)

        # (B, n_param_heads, N, q)
        param_embeddings = self.output_linear(param_embeddings)
        #with profiler.record_function("Get params"):
        couplings, fields = self._get_params_ardca(param_embeddings, N, padding_mask)

        return couplings, fields
    
    def sample_ardca(self, encodings, padding_mask, n_samples=1000):
        """Sampler for arDCA, currently works only for a single sequence.
            NB: This function should not be used for standard Potts."""
            ## Put model in evaluation mdoel
        B, N, _ = encodings.shape
        samples = torch.zeros(n_samples, N, dtype=torch.int)
        self.eval()
        q = self.q
        ## fields shape: (B,N,q), we will consider B=1 for the moment
        ## Couplings shape: (B, N*q, N*q)
        couplings, fields = self.forward_ardca(encodings, padding_mask)

        ## At the moment move to CPU! Then maybe move to GPU if we are able to vectorize
        couplings = couplings.to('cpu')
        fields = fields.to('cpu')
        ##############################################################################
        
        fields = fields[0,:].reshape(N, q)
        p_pos = softmax(-fields[0], dim=0)
        
        samples[:,0] = Categorical(p_pos).sample((n_samples,))
        Ham = torch.zeros(q)
        for sam in range(n_samples):
            print(f"We are at sample {sam} out of {n_samples}", end="\r")
            for pos in range(1,N):
                Ham[:] = fields[pos, :]
                for acc in range(pos):
                    for aa in range(q):
                        Ham[aa] += couplings[0, pos*q+aa, acc*q + samples[sam,acc]]
                
                p_pos[:] = softmax(-Ham, dim=0)
                samples[sam, pos] = Categorical(p_pos).sample()
        return samples