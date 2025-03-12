import torch
import torch.backends.mps
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from dummy_transformer_block import TransformerBlock,LayerNorm
import tiktoken 
from gpt_utils import *
from pretrained_utils import *

class DummyGPT(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        ##First Block
        self.tok_embed = nn.Embedding(cfg["vocab_size"],cfg["embed_size"])
        self.pos_embed = nn.Embedding(cfg["context_length"],cfg["embed_size"])
        self.drop_layer = nn.Dropout(cfg["dropout_rate"])
        #Second Block
        self.transformer_block =nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        #Last Block 
        self.out_norm_layer = LayerNorm(cfg["embed_size"])
        self.out_proj_layer = nn.Linear(cfg["embed_size"],cfg["vocab_size"],bias = False)

    def forward(self,in_ids):
        _,context_len = in_ids.shape
        tok_emds = self.tok_embed(in_ids) # (num_tokens,embed_size)
        pos_emds = self.pos_embed(torch.arange(context_len,device = in_ids.device))
        X = tok_emds + pos_emds
        X = self.drop_layer(X)
        X = self.transformer_block(X) #(num_tokens,embed_size)
        X = self.out_norm_layer(X)
        logits = self.out_proj_layer(X) #(num_tokens,vocab_size)

        return logits
        



class GPTDataset(Dataset):
    def __init__(self,text,tokenizer,context_length,stride):
        self.input_ids = []
        self.target_ids =[]

        token_ids = tokenizer.encode(text,allowed_special={"<|endoftext|>"})
        for i in range(0,len(token_ids) - context_length,stride):
            input_chunk = token_ids[i:i+context_length]
            target_chunk = token_ids[i+1:i+context_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx],self.target_ids[idx]
    
def createDataLoader(data,tokenizer,batch_size,context_length,stride,shuffle,drop_last=True,num_workers = 0):
    dataset = GPTDataset(data,tokenizer,context_length,stride)
    data_loader = DataLoader(dataset,batch_size,shuffle,drop_last=drop_last,num_workers=num_workers)

    return data_loader


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel() # Returns the total number of elements (or tokens) in the input_batch.
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0: 
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


       


          

## Test 
CFG = {
        "vocab_size":50257,
        "context_length":1024,
        "embed_size":768,
        "n_heads":12,
        "n_layers":12,
        "dropout_rate":0.1,
        "qkv_bias":True
    }

DATA__LOADER_CFG={
    "batch_size":32,
    "num_workers":0,
    "shuffle": True,
    "drop_last":True
}
with open("data/the_verdict.txt","r",encoding="UTF-8") as f:
    text = f.read()

train_ratio = 0.9
split_index = int(len(text) * 0.9)
train_data = text[:split_index]
val_data  = text[split_index:]
tokenizer = tiktoken.get_encoding("gpt2")
train_data_loader = createDataLoader(train_data,tokenizer,
                                    DATA__LOADER_CFG["batch_size"],
                                    CFG["context_length"],
                                    CFG["context_length"],
                                    DATA__LOADER_CFG['shuffle'],
                                    DATA__LOADER_CFG["drop_last"],
                                    DATA__LOADER_CFG['num_workers']
                                     )

val_data_loader = createDataLoader(val_data,tokenizer,
                                    DATA__LOADER_CFG["batch_size"],
                                    CFG["context_length"],
                                    CFG["context_length"],
                                   False,
                                    DATA__LOADER_CFG["drop_last"],
                                    DATA__LOADER_CFG['num_workers'] 
                                )


