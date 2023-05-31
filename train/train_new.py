import sys
sys.path.insert(1, "./../util/")
sys.path.insert(1, "./../model/")
from encoded_protein_dataset_new import get_embedding, EncodedProteinDataset_new, collate_fn_new
from pseudolikelihood import get_npll, get_npll_new, get_npll2
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
    default='/media/luchinoprince/b1715ef3-045d-4bdf-b216-c211472fb5a2/Data/InverseFolding/msas/', 
    help="directory with the Multiple Sequence Alignments(MSA) for both training and testing")
parser.add_argument('--encoding_dir', type=str,
    default='/media/luchinoprince/b1715ef3-045d-4bdf-b216-c211472fb5a2/Data/InverseFolding/structure_encodings/',
    help="Directory where to find all the encoded structures to pass to the Decoder for both training and testing")
parser.add_argument('--output_dir', type=str,
    default='/media/luchinoprince/b1715ef3-045d-4bdf-b216-c211472fb5a2/Data/InverseFolding/Intermediate_Models/',
    help="Directory where to store the benchmarking models and states. The frequence with with we do this depends on the bk_iter argument")
parser.add_argument('--device', type=int, default=0, help="GPU where we run our code", choices=[0, 1])
parser.add_argument('--seed', type=int, default=82477, 
    help="Random seed utlized for training")
parser.add_argument('--batch_size', type=int, default=32,
    help="Batch size used in the training/testing of the model")
parser.add_argument('--lr', type=float, default=1e-4,
    help="Learning rate of the Gradient Descent algorithm chosen")
parser.add_argument('--epochs', type=int, default=1000,
    help="The number of epochs done in the training procedure")
parser.add_argument('--test_epochs', type=int, default=10,
    help="Sets how often we calculate the negative pseudo-log-likelihoods for the different test datasets to monitor overfitting ")
parser.add_argument('--bk', type=bool, default=False, 
    help="Tells the program if we want to save intermediate models during the training")
parser.add_argument('--bk_epochs', type=int, default=50,
    help="How often we save intermediate models and states during the training procedure")
parser.add_argument('--dropout', type=float, default=0.1,
    help="Value of dropout in the attention layers of the decoder")
parser.add_argument('--eta', type=float, default=1e-4,
    help="Multiplicative factor in front of the L2 penalized negative pseudo-log-likelihood")
parser.add_argument('--noise', type=float, default=0.02, 
    help="This variable controls if we add noise to the encodings before training. The noise will be Gaussian with mean 0 and std=0.02, which corresponds roughly to 5-percent of the std we observe in the encodings")


args = parser.parse_args()

if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

### Setting up TensorBoard ######
#
#logdir = os.path.join('./runs', tb_folder)
summary_writer = SummaryWriter()
layout = {
    "metrics": {
        "loss": ["Multiline", ["loss/train", "loss/sequence", "loss/structure", "loss/superfamily"]],}
}
summary_writer.add_custom_scalars(layout)
print(f"The arguments of the experiment are:{args}")
print("I am loading the train and test datasets")
max_msas = args.max_msas
msa_dir = args.msa_dir
encoding_dir = args.encoding_dir
train_dataset = EncodedProteinDataset_new(os.path.join(msa_dir, 'train'), encoding_dir, noise=0.02, max_msas=max_msas)          ## Default value of noise used
sequence_test_dataset = EncodedProteinDataset_new(os.path.join(msa_dir, 'test/sequence'), encoding_dir, noise=0.0, max_msas=max_msas)
structure_test_dataset = EncodedProteinDataset_new(os.path.join(msa_dir, 'test/structure'), encoding_dir, noise=0.0, max_msas=max_msas)
superfamily_test_dataset = EncodedProteinDataset_new(os.path.join(msa_dir, 'test/superfamily'), encoding_dir, noise=0.0, max_msas=max_msas)

print(f"I have loaded the train and test datasets: seq:{len(sequence_test_dataset)}, str:{len(structure_test_dataset)}, super:{len(superfamily_test_dataset)}")

batch_structure_size = args.batch_size   ### I think with empty GPU we can hgo up to 10
perc_subset_test = 1.0     ## During the training, for every dataset available we select a random 10% of its samples
batch_msa_size = 16
q = 21                      ##isn't always 21??
collate_fn = partial(collate_fn_new, q=q, batch_msa_size=batch_msa_size)

train_loader = DataLoader(train_dataset, batch_size=batch_structure_size, collate_fn=collate_fn, shuffle=True)#, num_workers=3)
sequence_test_loader = DataLoader(sequence_test_dataset, batch_size=batch_structure_size, collate_fn=collate_fn, shuffle=False, 
sampler=RandomSampler(sequence_test_dataset, replacement=True, num_samples=int(perc_subset_test*len(sequence_test_dataset)/10)))#, num_workers=3)

structure_test_loader = DataLoader(structure_test_dataset, batch_size=batch_structure_size, collate_fn=collate_fn, shuffle=False, 
sampler=RandomSampler(structure_test_dataset, replacement=True, num_samples=int(perc_subset_test*len(structure_test_dataset))))#, num_workers=3)

superfamily_test_loader = DataLoader(superfamily_test_dataset, batch_size=batch_structure_size, collate_fn=collate_fn, shuffle=False, 
sampler=RandomSampler(superfamily_test_dataset, replacement=True, num_samples=int(perc_subset_test*len(superfamily_test_dataset))))#, num_workers=3)


decoder = None
embedding = None
torch.cuda.empty_cache()

seed = args.seed
torch.random.manual_seed(seed)
np.random.seed(seed)



epochs = args.epochs                                        ##Usual values are update steps=10^5, test_steps=10^2
test_epochs = args.test_epochs
bk_epochs = args.bk_epochs                                                  ## This tells us how ofter we save a model(default values is every ten-thousand updates)
update_steps = epochs * (len(train_dataset)//batch_structure_size + 1)   ## the other update steps will be used for "partial epochs", I want to save the last complet epoch
print(f"With update_steps:{update_steps} we will do {epochs} full epochs")

input_encoding_dim = 512
param_embed_dim = 512
n_param_heads = 52
d_model = 512
n_heads = 8
n_layers = 6
## Check before running which is the GPU which is free the most and put it as the running device
device = args.device
eta = args.eta
dropout = args.dropout

decoder = PottsDecoder(q, n_layers, d_model, input_encoding_dim, param_embed_dim, n_heads, n_param_heads, dropout=dropout)
decoder.to(device)
embedding = get_embedding(q)
embedding.to(device)

hyperparams = { 'lr':1e-4, 'eta':1e-4, 'batch_size':batch_structure_size, 'n_param_heads':n_param_heads, 'n_layers':n_layers, 
                'dropout':args.dropout, 'param_embed_dim':param_embed_dim, 'n_heads': n_heads}

optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr)




def get_loss_new(decoder, inputs, eta):
    """eta is the multiplicative term in front of the penalized negative pseudo-log-likelihood"""
    msas, encodings, padding_mask  = [input.to(device) for input in inputs]
    B, M, N = msas.shape
    #print(f"encodings' shape{encodings.shape}, padding mask:{padding_mask.shape}")
    param_embeddings, fields = decoder.forward_new(encodings, padding_mask)
    msas_embedded = embedding(msas)

    # get npll
    npll = get_npll2(msas_embedded, param_embeddings, fields, N, q)
    padding_mask_inv = (~padding_mask)
    # multiply with the padding mask to filter non-existing residues (this is probably not necessary)       
    npll = npll * padding_mask_inv.unsqueeze(1)
    npll_mean = torch.sum(npll) / (M * torch.sum(padding_mask_inv))
    
    Q = torch.einsum('bkuia, buhia->bkhia', 
                param_embeddings.unsqueeze(2), param_embeddings.unsqueeze(1)).sum(axis=-1)
    penalty = eta*(torch.sum(torch.sum(Q,axis=-1)**2) - torch.sum(Q**2) + torch.sum(fields**2))/B
    loss_penalty = npll_mean + penalty
    return loss_penalty, npll_mean.item() 

def get_loss_loader(decoder, loader, eta):

    decoder.eval()
    losses = 0
    iterator = 0
    with torch.no_grad():
        for inputs in loader:
            iterator+=1
            _, npll = get_loss_new(decoder, inputs, eta) 
            losses+=npll
    
    return losses/iterator


with tqdm(total = update_steps) as pbar: ##This is used to have the nice loading bar while training
    train_loss = 0
    update_step = 0
    #update_step=0
    bk_dir = args.output_dir       ## Folder to where we save the intermediate models
    train_batch_losses = []
    epoch = 0.0
    while update_step < update_steps:
        #decoder.train()
        for inputs in train_loader:
            decoder.train()
            loss_penalty, train_batch_loss = get_loss_new(decoder, inputs, eta)    ## get the current loss for the batch
            optimizer.zero_grad()                           ## set previous gradients to 0
            loss_penalty.backward()                         ## Get gradients
            loss_penalty.detach()
            optimizer.step()                                ## Do a step of GD
            update_step += 1                                ## Increase update step (the update steps will count also different batches within the same epoch)
            epoch = update_step / len(train_loader)
            train_batch_losses.append(train_batch_loss) ## Here we append the lossess in the different batches within the same epoch

            #################### BENCHMARK MODEL EVERY 50 EPOCHS ##########################################
            if args.bk and epoch % bk_epochs == 0:
                bk_dir= 'args.output_dir'
                fname_par = 'model_20_01_2023_epoch_' + str(epoch) + '.pt'

                ##Arguments of the model, could be inferred
                args_run = {}
                args_run['n_layers'] = n_layers
                args_run['input_encoding_dim'] = input_encoding_dim
                args_run['param_embed_dim'] = param_embed_dim
                args_run['n_heads'] = n_heads
                args_run['n_param_heads'] = n_param_heads
                args_run['dropout'] = args.dropout



                d = {}
                d['epoch'] = epoch
                d['update_step'] = update_step
                d['batch_size'] = batch_structure_size
                d['seed'] = seed
                d['eta'] = eta
                d['noise'] = args.noise
                d['args_run'] = args_run
                d['model_state_dict'] = decoder.state_dict()
                d['optimizer_state_dict'] = optimizer.state_dict()

                torch.save(d, os.path.join(bk_dir, fname_par))
            
            if (update_step  == 1) or (epoch % args.test_epochs == 0):                ## Don't we do this way the mean trough all the different batches and not within the same epoch? I think we would want the latter. Yes, infact we do it at the end of the if 
                ## We give the results not at every epoch, but at every "test_loss" update steps
                train_loss = np.mean(train_batch_losses)
                summary_writer.add_scalar('loss/train', train_loss, update_step)
                
                ## Lossess for the different test sets, want to use a subset of this only. Also want to pass only a random subset of it if possible
                sequence_test_loss = get_loss_loader(decoder, sequence_test_loader, eta)
                structure_test_loss = get_loss_loader(decoder, structure_test_loader, eta)
                superfamily_test_loss = get_loss_loader(decoder, superfamily_test_loader, eta)

                summary_writer.add_scalar('loss/sequence', sequence_test_loss, update_step)
                summary_writer.add_scalar('loss/structure', structure_test_loss, update_step)
                summary_writer.add_scalar('loss/superfamily', superfamily_test_loss, update_step)
                train_batch_losses = []
            
            ################################# SAVE FINAL MODEL ##############################################
            if update_step >= update_steps:
                bk_dir= 'D:/Data/InverseFolding/Intermediate_Models/'
                fname_par = 'model_20_01_2023_epoch_' + str(epoch) + '.pt'

                ##Arguments of the model, could be inferred
                args_run = {}
                args_run['n_layers'] = n_layers
                args_run['input_encoding_dim'] = input_encoding_dim
                args_run['param_embed_dim'] = param_embed_dim
                args_run['n_heads'] = n_heads
                args_run['n_param_heads'] = n_param_heads
                args_run['dropout'] = args.dropout



                d = {}
                d['epoch'] = epoch
                d['update_step'] = update_step
                d['batch_size'] = batch_structure_size
                d['seed'] = seed
                d['eta'] = eta
                d['noise'] = args.noise
                d['args_run'] = args_run
                d['model_state_dict'] = decoder.state_dict()
                d['optimizer_state_dict'] = optimizer.state_dict()

                torch.save(d, os.path.join(bk_dir, fname_par))
                break
            ##This gives me problem, I had to change the if condition
            pbar.set_description(f'update_step: {update_step}, epoch: {epoch:.2f}  train: {train_loss:.2f}, sequence: {sequence_test_loss:.2f}, structure: {structure_test_loss:.2f}, superfamily: {superfamily_test_loss:.2f}')
            pbar.update(1)

save_metrics = {'loss/train': train_loss, 'loss/sequence': sequence_test_loss, 
'loss/structure': structure_test_loss, 'loss/superfamily': superfamily_test_loss}
summary_writer.add_hparams(hyperparams, save_metrics)
summary_writer.close()
