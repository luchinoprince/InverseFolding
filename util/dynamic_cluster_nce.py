#### We are going to create the dynamic loader for NCE now

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence


############################################################################################
###################################### DYNAMIC LOADER ######################################   
############################################################################################
def dynamic_collate_fn(batch, q, batch_size, batch_msa_size, max_units = 12228):
    """ Dynamic Collate function for data loader, 2*8192 is 512*(16*2), which was the maximum number of input dimension for batch size equal to 16
    """
    batch_size = min(len(batch), batch_size)
    ## We selec a random subsample of the MSA
    msas = [tuple[0][torch.randint(0, tuple[0].shape[0], (batch_msa_size, )), :] for tuple in batch]

    # padding works in the second dimension
    msas = [torch.transpose(msa, 1, 0) for msa in msas]
    encodings = [tuple[1] for tuple in batch]
    pi_s = [tuple[2] for tuple in batch]

    inputs_packed = dynamic_cluster(batch_size, q, msas, encodings, pi_s, max_units=max_units)
    return batch_size, inputs_packed



def dynamic_cluster(batch_size, q, msas, encodings, pi_s, max_units):
    inputs_packed = []
    Ns = np.array([encoding.shape[0] for encoding in encodings])
    order = np.argsort(-Ns)  ## Need to reverse the order
    iterator = 0
    ## Order the encodigns based on batch size, this should also allow to know in advance where to split!!! 
    while iterator < batch_size:
        current_encodings = []
        current_msas = []
        current_pi_s = []
        dim = order[iterator]
        mini_batch_size = int(np.floor(max_units/Ns[dim]))
        for _ in range(min(mini_batch_size, batch_size-iterator)):
            current_encodings.append(encodings[order[iterator]])
            current_msas.append(msas[order[iterator]])
            current_pi_s.append(pi_s[order[iterator]])
            iterator+=1
        
        #N = current_encodings.shape[1][0]
        msas_pad = pad_sequence(current_msas, batch_first=True, padding_value=q)
        encodings_pad = pad_sequence(current_encodings, batch_first=True, padding_value=0.0) 
        #N = encodings_pad.shape[1]
        pi_s_pad = pad_sequence(current_pi_s, batch_first=True, padding_value=1/q) ## We put 1/N for the sampler
        msas_pad = torch.transpose(msas_pad, 2, 1)
        padding_mask = msas_pad[:, 0, :] == q

        inputs_packed.append((msas_pad, encodings_pad, pi_s_pad, padding_mask))
    return inputs_packed