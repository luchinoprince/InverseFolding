########## Noise-contrastive estimation ########
## We need to add another parameter, the normalizing constant Z.

import torch
import sys
sys.path.insert(1, "./../util")
sys.path.insert(1, "./../model")
from torch.distributions import Categorical
from torch.nn import BCEWithLogitsLoss
from encoded_protein_dataset_new import get_embedding

def nce(V, fields, Z, padding_mask, msas_embedded, msas_ind_embedded, mask_noise, fi, N, M, q):
    ## Mask-noise splits the training and noise samples, it is constant
    ## Its size is: (B, 2*M)
    ## V : (B, K, N, q)
    ## Fields: (B, N, q)
    ## Z: (B, 1)
    ## fi: (B, N, q)
    ## msas_embedded: (B, M, N, q)
    ## Get the samples from the independent models. Output (B, M, N)
    ## Change, put this outside eventually, already pass the samples
    #embedding = get_embedding(q)
    #samples_ind = torch.transpose(sample_ind(fi, n_samples=M), 0,1) 
    #samples_ind = embedding(samples_ind)
    ## I concatenate across the dimension of the MSA
    samples_full = torch.concat([msas_embedded, msas_ind_embedded], axis=1)    
    ## Brute force, use in-built functions then
    #print("Altro: ", loglik_potts(V, fields, Z, samples_full))
    G = loglik_potts(V, fields, Z, samples_full) - loglik_indep(fi, samples_full, padding_mask)
    #h = 1/(1+torch.exp(-G))

    #nce = 1/(2*M) * torch.sum(torch.log(h[:, :, 0:M]) + torch.log(1-h[:, :, M:]))
    loss = BCEWithLogitsLoss(reduction='none')
    ## nce is (B, M) --> be carefull at padding
    nce_l = loss(G, mask_noise)
    ## Return a (B,1) tensor
    return torch.mean(nce_l, axis=1)

def nce_old(V, fields, Z, msas_embedded, msas_ind_embedded, mask_noise, fi, N, M, q):
    ## Mask-noise splits the training and noise samples, it is constant
    ## Its size is: (B, 2*M)
    ## V : (B, K, N, q)
    ## Fields: (B, N, q)
    ## Z: (B, 1)
    ## fi: (B, N, q)
    ## msas_embedded: (B, M, N, q)
    ## Get the samples from the independent models. Output (B, M, N)
    ## Change, put this outside eventually, already pass the samples
    #embedding = get_embedding(q)
    #samples_ind = torch.transpose(sample_ind(fi, n_samples=M), 0,1) 
    #samples_ind = embedding(samples_ind)
    ## I concatenate across the dimension of the MSA
    samples_full = torch.concat([msas_embedded, msas_ind_embedded], axis=1)    
    ## Brute force, use in-built functions then
    G = loglik_potts(V, fields, Z, samples_full) - loglik_indep2(fi, samples_full)#loglik_indep(fi, samples_full, padding_mask_inv)
    #h = 1/(1+torch.exp(-G))
    #nce = 1/(2*M) * torch.sum(torch.log(h[:, :, 0:M]) + torch.log(1-h[:, :, M:]))
    loss = BCEWithLogitsLoss(reduction='none')
    ## nce is (B, M) --> be carefull at padding
    nce_l = loss(G, mask_noise)
    ## Return a (B,1) tensor
    return torch.mean(nce_l, axis=1)

def loglik_potts(V, F, logZ, msas_embedded):
    ## V and F have already been padded
    ## msas_embedded: (B, M, N, q)
    ## V: (B, K, N, q)
    ## F: (B, N, q) of fields. I have already summed over the K-th dimension. Fields DON'T need re-engeneering
    ## Z: (B, ) normalization constant for different Potts models
    ## inv_padding_mask: (B, N)
    K = V.shape[1]
    B, M, _, _ = msas_embedded.shape

    ## (B, 1, K, M, q) * (B, M, 1, N, q) = (B, M, K, N, q)  ---> (B, M, K, N)
    V_data = torch.sum(torch.unsqueeze(V, dim=1) * 
                       torch.unsqueeze(msas_embedded, dim=2), axis=-1)

    ## (B, M, K, N)  ---> (B, M, K)
    S_k = torch.sum(V_data, axis=-1)
    H_k = torch.sum(V_data**2, axis=-1)

    ## (B, M, K)  ---> (B, M)
    E = torch.sum(S_k**2 - H_k, axis=-1)

    ## (B, 1, N, q) * (B, M, N, q) = (B, M, N, q) ---> (B, M, N)
    Fi_data = torch.sum(torch.unsqueeze(F, dim=1) * msas_embedded, axis=-1)
    
    ## Fs is the field component of the sequence s: (B, M, N) --> (B, M) 
    Fs = torch.sum(Fi_data, axis=-1)

    ## Sum of the Hamiltonians of the sequences in the MSA
    ## (B,M) + (B,M) = (B,M) ---> (B, 1)
    #Ham_s = torch.sum(0.5*E + Fs, axis=-1)
    Ham_s = 0.5*E + Fs
    ## I return the likelihood, not the negative likelihood
    #return -torch.log(Z.unsqueeze(-1)) - Ham_s 
    return -logZ.unsqueeze(-1) - Ham_s#, Ham_s

def loglik_indep(fi, msas_embedded, padding_mask):
    ## fi not padded, this because since then we take the log it would have been problematic
    ## fi is given by the data, it's not a parameter of the model
    ## msas_embedded: (B, M, N, q)  ## it has a Kronecker parametrization. 
    ## fi: (B, N, q) it tells the frequence of of amino acid q in position N for the different batches
    ## (B, 1, N, q) * (B, M, N, q) = (B, M, N, q) ---> (B, M, N)
    padding_mask_inv = (~padding_mask)
    lprobs_i = torch.log(torch.sum(torch.unsqueeze(fi, dim=1) * msas_embedded, axis=-1))

    #print("lprobs:", lprobs_i)
    #print("Fi: ", fi)
    #print("check:", torch.sum(fi, dim=-1))
    ## (B, M, N) ---> (B, M)
    ## We set to zero padded positions
    #print(f"lprobs_i dim: {lprobs_i.shape}, padding_mask dim:{padding_mask_inv.shape}")
    lprobs_i = lprobs_i * padding_mask_inv.unsqueeze(1)
    lprobs_i = torch.nan_to_num(lprobs_i)
    lprob = torch.sum(lprobs_i, axis=2)
    return lprob

def loglik_indep2(fi, msas_embedded):
    ## fi not padded, this because since then we take the log it would have been problematic
    ## fi is given by the data, it's not a parameter of the model
    ## msas_embedded: (B, M, N, q)  ## it has a Kronecker parametrization. 
    ## fi: (B, N, q) it tells the frequence of of amino acid q in position N for the different batches
    ## (B, 1, N, q) * (B, M, N, q) = (B, M, N, q) ---> (B, M, N)
    probs_i = torch.sum(torch.unsqueeze(fi, dim=1) * msas_embedded, axis=-1)
    ## (B, M, N) ---> (B, M)
    ## We set to zero padded positions
    #print(f"lprobs_i dim: {lprobs_i.shape}, padding_mask dim:{padding_mask_inv.shape}")
    lprob = torch.sum(torch.log(probs_i), axis=2)
    return lprob


## I now creathe function to sample from the fields, as bmDCA cannot sample from just fields(previously noted)
def sample_ind(fi, n_samples=10000):
    ## I suppose I have single site frequencies, batched
    ## fi : (B, N, q)
    B, N, q = fi.shape
    m = Categorical(probs=fi)
    msa = m.sample(sample_shape=(n_samples,))

    ## msa is (M, B, N)
    return msa