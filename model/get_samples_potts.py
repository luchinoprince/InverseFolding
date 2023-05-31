import torch, tqdm
import sys
sys.path.insert(1, "./../util")
sys.path.insert(1, "./../model")
from torch.distributions import Categorical
from torch.nn import BCEWithLogitsLoss
from encoded_protein_dataset_new import get_embedding



def diverge(decoder, param_embeddings, fields, sequences_original, embedding, padding_mask, steps, clone=True, progress_bar=False, return_energies=False):
        ### Sequences original: (B, M, N) I don't want the embedding.
        device = next(decoder.parameters()).device
        B,M,N = sequences_original.shape
        steps = N
        ## We need this for the output, I already save the mean
        #llik_original = torch.mean(loglik_potts(param_embeddings, fields, get_embedding(sequences_original)))

        ## This is important since we don't want to select padding positions due to batching.
        ## This tells me the true lenghts of the different proteins in the batch, padding is always at the the end.
        true_Ns = (N-1) - torch.sum(padding_mask, dim=1).unsqueeze(-1)
        #print("True_Ns:", true_Ns)
        #print(padding_mask)
        with torch.no_grad():
            it = tqdm(range(steps)) if progress_bar else range(steps)
            ## Number of MC steps
            for step in it:
                ## I choose some random positions to update in the multiple sequence alignment. 
                ## INDS; (B, M)
                ## I need inds to be int64 due to having to use torch.scatter
                ## This formatting ensures that I never select padding positions in the MCMC procedure(padding is at the end)
                inds = torch.round(true_Ns * torch.rand((1, M), device=device)).type(torch.int64).to(device)
                llik_old = loglik_potts(param_embeddings, fields, embedding(sequences_original))
                ## Extract the old values for later: (B,M,1) --> (B,M)
                aa_old = torch.gather(sequences_original, dim=2, index=inds.unsqueeze(-1)).squeeze(1).to(device)
                #print(aa_old.shape)
                ## Draw the next values: (B,M)
                aa_new = torch.randint(0, decoder.q, (B, M,), device=device).type(torch.int32)#.cuda()
                #print("Original type:", sequences_original.dtype)
                #print("aa new type:", aa_new.dtype)
                ## Create the new sequence with the changes to get the likelihoods, I overwrite to save memory.
                ## I can do this since all the necessary original information is stored in aa_old (xN times memory saving)
                #sequences_new = torch.scatter(sequences_original, 2, inds.unsqueeze(-1), aa_new.unsqueeze(-1))
                sequences_original.scatter_(2, inds.unsqueeze(-1), aa_new.unsqueeze(-1))
                llik_new = loglik_potts(param_embeddings, fields, embedding(sequences_original))
                
                ## Create Metropolis hasting mask to accept/reject samples and then the final values
                switch = (torch.rand((B,M), device=device) < torch.exp((llik_new-llik_old))).int()
                aa_new = (1-switch)*aa_old.squeeze(-1) + switch*aa_new
                ## I get the true new likelihood, not the proposed ones
                #llik_new = (1-switch)*llik_old + switch*llik_new
                sequences_original.scatter_(2, inds.unsqueeze(-1), aa_new.unsqueeze(-1))
            #energies = energies * (1 - switch) + switch * new_energies
            #    if progress_bar:
            #        it.set_description('mean lik new: {0:.3f}, acc: {1:.3f}'.format(torch.mean(llik_new), torch.mean(switch.float())))
        #if return_energies:
        #    return sequences, energies
        #else:
        #return sequences_original, torch.mean(llik_new)-llik_original ## Be carefull at how to normalize here
        #return torch.mean(llik_new)
        return sequences_original
    
def loglik_potts(V, F, msas_embedded):
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
    return -Ham_s#, Ham_s

