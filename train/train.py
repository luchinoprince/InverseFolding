from encoded_protein_dataset import EncodedProteinDataset, collate_fn, get_embedding
from pseudolikelihood import get_npll
import torch
import numpy as np
from potts_decoder import PottsDecoder
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from tqdm import tqdm
import argparse
import os


#if __name__ == '__main__':   #not sure I need this
parser = argparse.ArgumentParser(description="Code to train the Potts decoder")
parser.add_argument('--max_msas', type=int, default=None, help="With this code we can select a subgroup of the training dataset of the desired size")

parser.add_argument('--msa_dir', type=str,
    default='/Data/InverseFoldingData/msas', 
    help="directory with the Multiple Sequence Alignments(MSA) for both training and testing")
parser.add_argument('--encoding_dir', type=str,
    default='/Data/InverseFoldingData/structure_encodings',
    help="Directory where to find all the encoded structures to pass to the Decoder for both training and testing")
parser.add_argument('--output_dir', type=str,
    default='/Data/InverseFoldingData/IntermediateModels',
    help="Directory where to store the benchmarking models and states. The frequence with with we do this depends on the bk_iter argument")
parser.add_argument('--device', type=int, default=0, help="GPU where we run our code", choices=[0, 1, 2, 3])
parser.add_argument('--seed', type=int, default=0, 
    help="Random seed utlized for training")
parser.add_argument('--batch_size', type=int, default=8,
    help="Batch size used in the training/testing of the model")
parser.add_argument('--lr', type=float, default=0.0001,
    help="Learning rate of the Gradient Descent algorithm chosen")
parser.add_argument('--update_steps', type=int, default=1e5,
    help="The number of updating steps done in the training procedure")
parser.add_argument('--test_steps', type=int, default=1e3,
    help="Sets how often we calculate the negative pseudo-log-likelihoods for the different test datasets to monitor overfitting ")
parser.add_argument('--bk', type=bool, default=False, 
    help="Tells the program if we want to save intermediate models during the training")
parser.add_argument('--bk_iter', type=int, default=1e4,
    help="How often we save intermediate models and states during the training procedure")
parser.add_argument('--dropout', type=float, default=0.1,
    help="Value of dropout in the attention layers of the decoder")
parser.add_argument('--eta', type=float, default=1e-3,
    help="Multiplicative factor in front of the L2 penalized negative pseudo-log-likelihood")
parser.add_argument('--noise', type=float, default=0.02, 
    help="This variable controls if we add noise to the encodings before training. The noise will be Gaussian with mean 0 and std=0.02, which corresponds roughly to 5-percent of the std we observe in the encodings")


args = parser.parse_args()

if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

### Setting up TensorBoard ######
tb_folder = 'dropout_'+ str(args.dropout) + 'noise_' + str(args.noise) +'_new'
logdir = os.path.join('./runs', tb_folder)
summary_writer = SummaryWriter(log_dir=logdir)
layout = {
    "metrics": {
        "loss": ["Multiline", ["loss/train", "loss/sequence", "loss/structure", "loss/superfamily"]],}
}
summary_writer.add_custom_scalars(layout)

print(f"The arguments of the experiment are:{args}")
print("I am loading the train and test datasets")
train_dataset = EncodedProteinDataset(os.path.join(args.msa_dir, 'train'), args.encoding_dir, noise=args.noise, max_msas=args.max_msas)
sequence_test_dataset = EncodedProteinDataset(os.path.join(args.msa_dir, 'test/sequence'), args.encoding_dir, noise=0.0, max_msas=args.max_msas)
structure_test_dataset = EncodedProteinDataset(os.path.join(args.msa_dir, 'test/structure'), args.encoding_dir, noise=0.0, max_msas=args.max_msas)
superfamily_test_dataset = EncodedProteinDataset(os.path.join(args.msa_dir, 'test/superfamily'), args.encoding_dir, noise=0.0, max_msas=args.max_msas)
print(f"I have loaded the train and test datasets: seq:{len(sequence_test_dataset)}, str:{len(structure_test_dataset)}, super:{len(superfamily_test_dataset)}")

batch_structure_size = args.batch_size   ### I think with empty GPU we can hgo up to 10
perc_subset_test = 0.1     ## During the training, for every dataset available we select a random 10% of its samples
batch_msa_size = 16
q = 21                      ##isn't always 21??
collate_fn = partial(collate_fn, q=q, batch_msa_size=batch_msa_size)
train_loader = DataLoader(train_dataset, batch_size=batch_structure_size, collate_fn=collate_fn, shuffle=True)
sequence_test_loader = DataLoader(sequence_test_dataset, batch_size=batch_structure_size, collate_fn=collate_fn, shuffle=False, 
sampler=RandomSampler(sequence_test_dataset, replacement=True, num_samples=int(perc_subset_test*len(sequence_test_dataset)/10)))

structure_test_loader = DataLoader(structure_test_dataset, batch_size=batch_structure_size, collate_fn=collate_fn, shuffle=False, 
sampler=RandomSampler(structure_test_dataset, replacement=True, num_samples=int(perc_subset_test*len(structure_test_dataset))))

superfamily_test_loader = DataLoader(superfamily_test_dataset, batch_size=batch_structure_size, collate_fn=collate_fn, shuffle=False, 
sampler=RandomSampler(superfamily_test_dataset, replacement=True, num_samples=int(perc_subset_test*len(superfamily_test_dataset))))


decoder = None
embedding = None
torch.cuda.empty_cache()

seed = args.seed
torch.random.manual_seed(seed)
np.random.seed(seed)



update_steps = args.update_steps                                        ##Usual values are update steps=10^5, test_steps=10^2
test_steps = args.test_steps
bk_iter = args.bk_iter                                                  ## This tells us how ofter we save a model(default values is every ten-thousand updates)
n_epochs = update_steps//(len(train_dataset)//batch_structure_size)   ## the other update steps will be used for "partial epochs", I want to save the last complet epoch
print(f"With update_steps:{update_steps} we will do {n_epochs} full epochs")

input_encoding_dim = train_dataset.encoding_dim
param_embed_dim = 512
n_param_heads = 4
d_model = 128
n_heads = 2
n_layers = 2
## Check before running which is the GPU which is free the most and put it as the running device
device = args.device
eta = args.eta
dropout = args.dropout

decoder = PottsDecoder(q, n_layers, d_model, input_encoding_dim, param_embed_dim, n_heads, n_param_heads, dropout=dropout)
decoder.to(device)
embedding = get_embedding(q)
embedding.to(device)

#decoder.train()
#print(decoder.state_dict()['P'].requires_grad)

model_path = '/Data/InverseFoldingData/IntermediateModels/parameters_seed_0_batch_size_4_nheads_2_d_128_nparheads_4_dropout_0.1_eta_0.001_update_500000_noise_0.02'
opt_path = '/Data/InverseFoldingData/IntermediateModels/opt_state_seed_0_batch_size_4_nheads_2_d_128_nparheads_4_dropout_0.1_eta_0.001_update_500000_noise_0.02'
decoder.load_state_dict(torch.load(model_path))

optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
opt_params = torch.load(opt_path)
optimizer.state = opt_params



def get_loss(decoder, inputs, eta):
    """eta is the multiplicative term in front of the penalized negative pseudo-log-likelihood"""
    msas, encodings, padding_mask  = [input.to(device) for input in inputs]
    B, M, N = msas.shape
    #print(f"encodings' shape{encodings.shape}, padding mask:{padding_mask.shape}")
    couplings, fields = decoder(encodings, padding_mask)

    # embed and reshape to (B, M, N*q)
    msas_embedded = embedding(msas).view(B, M, -1)

    # get npll
    npll = get_npll(msas_embedded, couplings, fields, N, q)
    padding_mask_inv = (~padding_mask)

    # multiply with the padding mask to filter non-existing residues (this is probably not necessary)       
    npll = npll * padding_mask_inv.unsqueeze(1)
    penalty = eta*(torch.sum(couplings**2) + torch.sum(fields**2))/B

    # the padding mask does not contain the msa dimension so we need to multiply by M
    npll_mean = torch.sum(npll) / (M * torch.sum(padding_mask_inv))
    loss_penalty = npll_mean + penalty

    return loss_penalty, npll_mean.item()  ##we can just add the penalty since we have set already to 0 couplings and fields of padded elements

def get_loss_new(decoder, inputs, eta):
    """eta is the multiplicative term in front of the penalized negative pseudo-log-likelihood"""
    msas, encodings, padding_mask  = [input.to(device) for input in inputs]
    B, M, N = msas.shape
    #print(f"encodings' shape{encodings.shape}, padding mask:{padding_mask.shape}")
    param_embeddings, fields = decoder.forward_new(encodings, padding_mask)

    # get npll
    npll2 = get_npll2(msas_embedded, param_embeddings, fields, N, q)


    padding_mask_inv = (~padding_mask)
    # multiply with the padding mask to filter non-existing residues (this is probably not necessary)       
    npll = npll * padding_mask_inv.unsqueeze(1)
    
    ### We have to calculate the penalty, we want to use einsum
    ## (B, K, N, q) ---> (B, N, q, K)
    ## param_embeddings_t = torch.permute(param_embeddings, (0,2,3,1))
    
    ## (B, K, 1N, q) @ (B, 1, K, N, q) = (B, K, K, N, q) ---> (B, K, K)
    Q = torch.einsum('bkuia, buhia->bkhia', 
                     param_embeddings.unsqueeze(2), param_embeddings.unsqueeze(1)).sum(axis=-1).sum(axis=-1)
    
    ## Later implement with einsum
    ## First multiplication: (B, N, q, 1, K) * (B, 1, 1, K, K) = (B, N, q, 1, K)
    ## Second multiplication: (B, N, Q, 1, K) * (B, N, Q, K, 1) = (B, N, Q, 1, 1)
    ## Flatten + Sums: (B, N, Q, 1, 1) --> (B, N, Q) ---> (B)
    aux = torch.matmul(torch.matmul(param_embeddings.permute((0,2,3,1)).unsqueeze(-2), Q.unsqueeze(1).unsqueeze(1)),
                           param_embeddings.permute((0,2,3,1)).unsqueeze(-1)).flatten(-2, 1).sum(axis=-1).sum(axis=-1)
    ## Again we don't need any reegeneering for fields
    penalty = eta*(aux - (param_embeddings**2).sum(axis=1).sum(axis=-1).sum(axis=-1) + fields**2)/B            
    # the padding mask does not contain the msa dimension so we need to multiply by M
    npll_mean2 = torch.sum(npll2) / (M * torch.sum(padding_mask_inv))
    loss_penalty2 = npll_mean2 + penalty

    return loss_penalty2, npll_mean2.item()  

def get_loss_loader(decoder, loader, eta):

    decoder.eval()
    losses = []
    with torch.no_grad():
        for inputs in loader:
            _, npll = get_loss(decoder, inputs, eta) 
            losses.append(npll)
    
    return np.mean(losses)


with tqdm(total = update_steps) as pbar: ##This is used to have the nice loading bar while training
    train_loss = 0
    update_step = int(5*1e5)
    #update_step=0
    bk_dir = args.output_dir       ## Folder to where we save the intermediate models
    train_batch_losses = []
    epoch = 0.0
    while update_step < update_steps:
        #decoder.train()
        for inputs in train_loader:
            decoder.train()
            #for acc in range(2):                            ##To accumulate gradients without going CUDA out of memory
            #print(decoder.state_dict()['attention_layers.1.norm1.bias'])
            #print(torch.max(decoder.state_dict()['P']))
            #print(torch.min(decoder.state_dict()['P']))
            #print(optimizer.state)
            #print(decoder.P.requires_grad)
            #print(torch.max(decoder.P))
            #print(torch.min(decoder.P))
            #print(decoder.state_dict()['attention_layers.1.norm1.bias'].requires_grad)

            loss_penalty, train_batch_loss = get_loss(decoder, inputs, eta)    ## get the current loss for the batch
            optimizer.zero_grad()                           ## set previous gradients to 0
            loss_penalty.backward()                         ## Get gradients
            #print("This is the gradient: ", decoder.P.grad)
            #print(decoder.field_linear.weight.grad)
            loss_penalty.detach()
            optimizer.step()                                ## Do a step of GD
            update_step += 1                                ## Increase update step (the update steps will count also different batches within the same epoch)
            epoch = update_step / len(train_loader)

            if args.bk and update_step%bk_iter==0:
                fname_par = f"parameters_seed_{seed}_batch_size_{args.batch_size}_nheads_{n_heads}_d_{d_model}_nparheads_{n_param_heads}_dropout_{dropout}_eta_{eta}_update_{update_step}_noise_{args.noise}"
                fname_opt = f"opt_state_seed_{seed}_batch_size_{args.batch_size}_nheads_{n_heads}_d_{d_model}_nparheads_{n_param_heads}_dropout_{dropout}_eta_{eta}_update_{update_step}_noise_{args.noise}"
                torch.save(decoder.state_dict(), os.path.join(bk_dir, fname_par))
                torch.save(optimizer.state, os.path.join(bk_dir, fname_opt))

            train_batch_losses.append(train_batch_loss) ## Here we append the lossess in the different batches within the same epoch
            
            ## We want to keep track of the test loss not at every batch, too costrly otherwise. Usually set to once every 100.
            if (update_step==int(5*1e5) + 1 or update_step % test_steps == 0) or update_step == update_steps - 1:
                ## Don't we do this way the mean trough all the different batches and not within the same epoch? I think we would want the latter. Yes, infact we do it at the end of the if 
                ## We give the results not at every epoch, but at every "test_loss" update steps
                train_loss = np.mean(train_batch_losses)
                summary_writer.add_scalar('loss/train', train_loss, update_step)
                del loss_penalty
                del train_batch_losses
                
                ## Lossess for the different test sets, want to use a subset of this only. Also want to pass only a random subset of it if possible
                sequence_test_loss = get_loss_loader(decoder, sequence_test_loader, eta)
                structure_test_loss = get_loss_loader(decoder, structure_test_loader, eta)
                superfamily_test_loss = get_loss_loader(decoder, superfamily_test_loader, eta)

                summary_writer.add_scalar('loss/sequence', sequence_test_loss, update_step)
                summary_writer.add_scalar('loss/structure', structure_test_loss, update_step)
                summary_writer.add_scalar('loss/superfamily', superfamily_test_loss, update_step)
            
                
                ###already done this calculation, I think we can take it out.
                #train_loss = np.mean(train_batch_losses)  
                train_batch_losses = []
            
            if update_step >= update_steps:
                fname_par = f"parameters_seed_{seed}_batch_size_{args.batch_size}_nheads_{n_heads}_d_{d_model}_nparheads_{n_param_heads}_dropout_{dropout}_eta_{eta}_noise_{args.noise}_final"
                fname_opt = f"opt_state_seed_{seed}_batch_size_{args.batch_size}_nheads_{n_heads}_d_{d_model}_nparheads_{n_param_heads}_dropout_{dropout}_eta_{eta}_noise_{args.noise}_final"
                torch.save(decoder.state_dict(), os.path.join(bk_dir, fname_par))
                torch.save(optimizer.state, os.path.join(bk_dir, fname_opt))
                break
            ##This gives me problem, I had to change the if condition
            pbar.set_description(f'update_step: {update_step}, epoch: {epoch:.2f}  train: {train_loss:.2f}, sequence: {sequence_test_loss:.2f}, structure: {structure_test_loss:.2f}, superfamily: {superfamily_test_loss:.2f}')
            pbar.update(1)