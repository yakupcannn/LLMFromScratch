import tiktoken 
from torch.utils.data import Dataset,DataLoader
class GPTDataset(Dataset):
    def __init__(self,text,tokenizer,context_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text,allowed_special ={"<|endoftext|>"})

        for i in range(0,len(text),stride):
            input_chunk = token_ids[i:i+context_length]
            target_chunk = token_ids[i+1:i+1+context_length]
            self.input_ids.append(input_chunk)
            self.target_ids.append(target_chunk)

    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx],self.target_ids[idx]


def create_dataloader(text,batch_size = 4,context_length =256,stride =128,shuffle = True,drop_last = True,num_workers=0):
    tokenizer  = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(text,tokenizer,context_length,stride)
    data_loader = DataLoader(dataset,batch_size,shuffle,drop_last=drop_last,num_workers=num_workers)

    return data_loader


if __name__ == "__main__":
    with open("../data/the_verdict.txt","r",encoding="UTF-8") as f:
        raw_text = f.read()

    print(raw_text)


    loader = create_dataloader(raw_text,batch_size=1,context_length=4,stride = 1,shuffle=False)
    iter_loader = iter(loader)
    first_batch = next(iter_loader)
    print(first_batch)
    



   

        












