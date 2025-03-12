import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import tiktoken
from dummy_gpt import DummyGPT
from pretrained_utils import *
import torch.nn.functional as F
import matplotlib.pyplot as plt

class SpamDataset(Dataset):
    def __init__(self,csv_file,tokenizer,max_length = None,pad_token_id = 50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
        if max_length is None:
            self.max_length = self._find_longest_encoded_text()
        else:
            self.max_length = max_length
        # If the max_length is smaller than encoded_text length , truncate the encoded text
        self.encoded_texts = [encoded_text[:max_length] for encoded_text in self.encoded_texts]

        ## Add pad tokens
        self.encoded_texts  = [ encode_text+[pad_token_id]*(self.max_length - len(encode_text)) for encode_text in self.encoded_texts]

    def __getitem__(self, index):
        data = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return torch.tensor(data,dtype=torch.long),torch.tensor(label,dtype = torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def _find_longest_encoded_text(self):
        mx = 0 
        for encode_text in self.encoded_texts:
            if len(encode_text)> mx:
                mx = len(encode_text)
        return mx 


            
def extract_data_from_csv_file(data_file_path):
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    return df


def get_balanced_dataset(data):
    ham = data[data["Label"]=="ham"]
    spam = data[data["Label"] == "spam"]
    balance = len(spam) if len(ham)>len(spam) else len(ham)
    ham = ham[:balance]
    spam = spam[:balance]
    return pd.concat([ham,spam])
    

def get_train_val_test_dataset(dataset,train_ratio,val_ratio):
    train_dataset = dataset[:int(len(dataset)*train_ratio)]
    val_dataset = dataset[int(len(dataset)*train_ratio):int(len(dataset)*(train_ratio+val_ratio))]
    test_dataset = dataset[int(len(dataset)*(train_ratio+val_ratio)):] 

    return train_dataset,val_dataset,test_dataset

def train_model(train_loader,val_loader,model,optimizer,epochs):
    train_losses,val_losses = [],[]
    train_accs,val_accs= [],[]
    
    
    ## TRAINING LOSS
    model.train()
    for epoch in range(epochs):
        train_epoch_loss = 0.0 
        train_acc = 0.0
        for t_input_batch,t_target_batch in train_loader:
            t_outputs = model(t_input_batch)
            last_logits = t_outputs[:,-1,:]
            correct = (torch.argmax(last_logits,dim= -1) == t_target_batch).sum().item()
            train_acc += correct/ last_logits.shape[0]
            optimizer.zero_grad()
            train_loss = F.cross_entropy(last_logits,t_target_batch)
            train_epoch_loss+=train_loss.item()
            train_loss.backward()
            optimizer.step()
        train_accs.append(train_acc/len(train_loader))
        avg_train_loss = train_epoch_loss/len(train_loader)
        train_losses.append(avg_train_loss)
        
        ## VALIDATION LOSS
        model.eval()  # Set the model to evaluation mode
        val_epoch_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for v_input_batch, v_target_batch in val_loader:
                v_outputs = model(v_input_batch)
                last_logits = v_outputs[:,-1,:]
                correct = (torch.argmax(last_logits,dim= -1) == v_target_batch).sum().item()
                val_acc += correct/ last_logits.shape[0]
                val_loss = F.cross_entropy(last_logits, v_target_batch)
                val_epoch_loss += val_loss.item()
        val_accs.append(val_acc/len(val_loader))
        avg_val_loss = val_epoch_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Validation Loss: {avg_val_loss:.4f} | ")
    
    return (train_losses,val_losses),(train_accs,val_accs)



def classify_text(text,model,tokenizer,device,max_length = None,pad_token_id = 50256):
    model.eval()
    input_ids = tokenizer.encode(text)
    context_length = model.pos_embed.weight.shape[0]
    if max_length is None:
        max_length = len(input_ids)
    input_ids = input_ids[:min(context_length,max_length)]
    input_ids += [pad_token_id] *(max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids,device = device).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)[:,-1,:]
        class_label = torch.argmax(outputs,dim = -1,keepdim=True)
        return "Spam" if class_label.item() == 1  else "Not Spam"


def evaluate_test_scores(test_loader,model,device,epochs):
    model.eval()
    test_accs = []
    test_losses = []
    for epoch in range(epochs):
        test_acc = 0.0
        test_epoch_loss = 0.0
        for input_batch,target_batch in test_loader:
            with torch.no_grad():
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)
                outputs = model(input_batch)[:,-1,:]
                correct = (torch.argmax(outputs,dim= -1) == target_batch).sum().item()
                test_acc += correct/ outputs.shape[0]
                test_loss = F.cross_entropy(outputs, target_batch)
                test_epoch_loss += test_loss.item()
        test_accs.append(test_acc/len(test_loader))
        avg_test_loss = test_epoch_loss / len(test_loader)
        test_losses.append(avg_test_loss)
    return test_losses,test_accs

                





        






## MAIN 
df = extract_data_from_csv_file("data/sms+spam+collection/SMSSpamCollection")
df["Label"].value_counts()

balanced_dataset = get_balanced_dataset(df)
balanced_dataset=balanced_dataset.sample(len(balanced_dataset),random_state=123)
balanced_dataset["Label"] = balanced_dataset["Label"].map({"ham": 0, "spam": 1})
train_ratio = 0.7
val_ratio = 0.1
train_dataset,val_dataset,test_dataset = get_train_val_test_dataset(balanced_dataset,train_ratio,val_ratio)
train_dataset.to_csv("data/spam_data/train_data.csv",index=False)
val_dataset.to_csv("data/spam_data/val_data.csv",index=False)
test_dataset.to_csv("data/spam_data/test_data.csv",index = False)

tokenizer = tiktoken.get_encoding("gpt2")
train_dataset = SpamDataset("data/spam_data/train_data.csv",tokenizer)
val_dataset = SpamDataset("data/spam_data/val_data.csv",tokenizer,max_length=train_dataset.max_length)
test_dataset = SpamDataset("data/spam_data/test_data.csv",tokenizer,max_length=train_dataset.max_length)

MODEL_CFG = {
    "BATCH_SIZE" :8,
    "N_WORKERS":0,
    "LR":5e-5,
    "N_EPOCHS" : 5
}
train_loader = DataLoader(
    dataset = train_dataset,
    batch_size= MODEL_CFG["BATCH_SIZE"],
    num_workers=MODEL_CFG['N_WORKERS'],
    drop_last= True
)
val_loader = DataLoader(
    dataset = val_dataset,
    batch_size= MODEL_CFG["BATCH_SIZE"],
    num_workers=MODEL_CFG['N_WORKERS'],
    drop_last= False
)
test_loader = DataLoader(
    dataset = test_dataset,
    batch_size= MODEL_CFG["BATCH_SIZE"],
    num_workers=MODEL_CFG['N_WORKERS'],
    drop_last= False
)

CFG = {
        "vocab_size":50257,
        "context_length":1024,
        "embed_size":768,
        "n_heads":12,
        "n_layers":12,
        "dropout_rate":0.0,
        "qkv_bias":True,
        "n_classes":2
    }
pretrained_gpt = DummyGPT(CFG)
MODEL_DIR = "GPT_2_Weights/124M/"

settings,tf_ckpt_path = load_model_params(MODEL_DIR)
gpt_params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path,settings)
load_gpt_params_into_custom_model(pretrained_gpt,gpt_params)

print(pretrained_gpt)
# FREEZE ALL PARAMETERS
for parameter in pretrained_gpt.parameters():
    parameter.requires_grad = False

# UNFREEZE NECESSARY PARAMETERS 
pretrained_gpt.out_proj_layer = torch.nn.Linear(in_features=CFG["embed_size"],out_features=CFG["n_classes"])
#Last Transformer Block
for param in pretrained_gpt.transformer_block[-1].parameters():
    param.requires_grad = True
# Output Norm Layer
for param in pretrained_gpt.out_norm_layer.parameters():
    param.requires_grad = True


optimizer = torch.optim.AdamW(params=pretrained_gpt.parameters(),
                              lr = MODEL_CFG["LR"],weight_decay=0.1)

(train_losses,val_losses),(train_accs,val_accs) = train_model(train_loader,val_loader,pretrained_gpt,optimizer,MODEL_CFG["N_EPOCHS"])


## Plot Losses
plt.figure(figsize=(10, 6))

# Plotting training, validation, and test losses
plt.plot(train_losses, label="Train Loss", color="blue")
plt.plot(val_losses, label="Validation Loss", color="green")

# Adding labels and title
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training And Validation Losses per Epoch")
plt.legend()

# Show the plot
plt.show()

plt.figure(figsize=(10, 6))

# Plotting training, validation, and test accuracies
plt.plot(train_accs, label="Train Accuracy", color="blue")
plt.plot(val_accs, label="Validation Accuracy", color="green")

# Adding labels and title
plt.xlabel("Epochs")
plt.ylabel("Accuracies")
plt.ylim(0,1)
plt.title("Training And Validation Accuracy per Epoch")
plt.legend()

# Show the plot
plt.show()

torch.save(pretrained_gpt.state_dict(),"./model/pretrained_gpt_v4.pth")

##Load pretrained model 

pretrained_state_dict = torch.load("model/pretrained_gpt_v4.pth")

pretrained_gpt.load_state_dict(pretrained_state_dict)
text = "FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv"

result = classify_text(text,pretrained_gpt,tokenizer,"cpu",max_length=train_dataset.max_length)
print(result)


## Metrics 

avg_val_acc = 0.0
for val in val_accs:
    avg_val_acc += val
avg_train_acc = 0.0
for train in train_accs:
    avg_train_acc += train
print(f"Average Validation Accuracy:{avg_val_acc/len(val_accs)}")
print(f"Average Train Accuracy:{avg_train_acc/len(train_accs)}")

##Evaluate All Test Dataset
test_losses, test_accs = evaluate_test_scores(test_loader,pretrained_gpt,"cpu",MODEL_CFG["N_EPOCHS"])
avg_test_loss = 0.0
avg_test_acc = 0.0
for test_acc in test_accs:
    avg_test_acc += test_acc
for test_loss in test_losses:
    avg_test_loss += test_loss
avg_test_acc /= len(test_accs)
avg_test_loss /= len(test_losses)
print(f"Average Test Accuracy:{avg_test_acc:.2f} | Average Test Loss:{avg_test_loss:.2f}")

