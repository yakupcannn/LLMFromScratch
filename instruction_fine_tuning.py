import json
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import tiktoken
from pretrained_utils import *
from dummy_gpt import DummyGPT,train_model_simple
from gpt_utils import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator



class InstructionFTDataset(Dataset):
    def __init__(self,data,tokenizer):
        self.data = data
        self.encoded_data = []
        for entry in data:
            instruction_input = alpaca_format_input(entry)
            response = f"\n\n### Response:\n{entry['output']}"
            instruction_data = instruction_input + response
            self.encoded_data.append(tokenizer.encode(instruction_data))

    def __getitem__(self, index):
        return self.encoded_data[index]
    
    def __len__(self):
        return len(self.encoded_data)

def prepare_input_target(batch,pad_token_id=50256,max_context_length = 1024,device="cpu",ignore_index = -100):
    batch_max_length = max(len(data)+1 for data in batch)
    input_dataset = []
    target_dataset = []
    for data in batch:
        n_pad = batch_max_length - len(data)
        new_data = data.copy()
        new_data += [pad_token_id]
        target_data  = new_data[1:]
        new_data += [pad_token_id] * (n_pad -1)
        target_data +=[ignore_index] * (n_pad -1) 
        input_batch = torch.tensor(new_data[:-1])
        target_batch = torch.tensor(target_data)
        if max_context_length is not None and max_context_length <= 1024 and max_context_length<(batch_max_length-1):
            input_batch = input_batch[:max_context_length]
            target_batch = target_batch[:max_context_length]

        input_dataset.append(input_batch)
        target_dataset.append(target_batch)

       

        inputs = torch.stack(input_dataset).to(device)
        targets = torch.stack(target_dataset).to(device)
    return (inputs,targets)

    

def read_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def alpaca_format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text

def train_test_split(data,train_ratio,test_ratio,shuffle = True):
    if shuffle:
        np.random.shuffle(data)
    
    train_split = int(len(data) *train_ratio)
    test_split = int(len(data)*test_ratio)
    train_data = data[:train_split]
    test_data = data[train_split:train_split+test_split]
    val_data = data[train_split+test_split:]

    return train_data,test_data,val_data


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.show()
    
##MAIN 
MODEL_CFG = {
    "BATCH_SIZE":8,
    "LR":0.000001,
    "N_WORKERS":0,
    "W_DECAY":0.01,
    "N_EPOCHS":1
}
tokenizer = tiktoken.get_encoding("gpt2")
FILE_PATH = "data/instruction_data/text_data.json"
data = read_data(FILE_PATH)
input = alpaca_format_input(data[0])
target_response = f"\n\n### Response:\n{data[0]['output']}"


TRAIN_RATIO = 0.85
TEST_RATIO = 0.10
train_data,test_data,val_data = train_test_split(data,TRAIN_RATIO,TEST_RATIO,shuffle=True)


train_dataset = InstructionFTDataset(train_data,tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size= MODEL_CFG["BATCH_SIZE"],
    shuffle=True,
    collate_fn= prepare_input_target,
    drop_last=True,
    num_workers=MODEL_CFG["N_WORKERS"]
)
test_dataset = InstructionFTDataset(test_data,tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size= MODEL_CFG["BATCH_SIZE"],
    shuffle=False,
    collate_fn= prepare_input_target,
    drop_last=False,
    num_workers=MODEL_CFG["N_WORKERS"]
)
val_dataset = InstructionFTDataset(val_data,tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size= MODEL_CFG["BATCH_SIZE"],
    shuffle=False,
    collate_fn= prepare_input_target,
    drop_last=False,
    num_workers=MODEL_CFG["N_WORKERS"]
)

BASE_CONFIG = {
"vocab_size": 50257,     # Vocabulary size
"context_length": 1024,  # Context length
"dropout_rate": 0.0,        # Dropout rate
"qkv_bias": True         # Query-key-value bias
}
model_configs = {
"gpt2-small (124M)": {"embed_size": 768, "n_layers": 12, "n_heads": 12},
"gpt2-medium (355M)": {"embed_size": 1024, "n_layers": 24, "n_heads": 16},
"gpt2-large (774M)": {"embed_size": 1280, "n_layers": 36, "n_heads": 20},
"gpt2-xl (1558M)": {"embed_size": 1600, "n_layers": 48, "n_heads": 25},
}
CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
device = torch.device("cpu")
MODEL_DIR = "GPT_2_Weights/355M/"
settings,tf_ckpt_path = load_model_params(MODEL_DIR)
gpt_params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path,settings)
instruction_model = DummyGPT(BASE_CONFIG)
load_gpt_params_into_custom_model(instruction_model,gpt_params)
    
## FINE TUNING 
torch.manual_seed(123)
optimizer = torch.optim.AdamW(instruction_model.parameters(), lr=MODEL_CFG["LR"], weight_decay=MODEL_CFG["W_DECAY"])
train_losses, val_losses, tokens_seen = train_model_simple(
    instruction_model, train_loader, val_loader, optimizer, device,
    num_epochs=MODEL_CFG["N_EPOCHS"], eval_freq=5, eval_iter=5,
    start_context=alpaca_format_input(val_data[0]), tokenizer=tokenizer
)
epochs_tensor = torch.linspace(0, MODEL_CFG["N_EPOCHS"], len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
torch.save(instruction_model.state_dict(),"model/instruction/instruction_v4.pth")


ins_state_dict = torch.load("model/instruction/instruction_v4.pth")
instruction_model.load_state_dict(ins_state_dict)

for data in test_data[20:40]:
    alpaca_input = alpaca_format_input(data)
    X_ids = encode_text_token(alpaca_input,tokenizer)
    next_token_ids = generate_next_words(instruction_model,X_ids,256,BASE_CONFIG["context_length"],eos_id=50256)
    generated_text = decode_token_ids(next_token_ids,tokenizer)
    response_text = generated_text[len(alpaca_input):].replace("### Response:","").replace("\n","")

    print(alpaca_input)
    print(f"Correct Response: {data['output']}")
    print(f"Model Guess: {response_text}")
    print("="*20)
