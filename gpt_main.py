import torch
from dummy_gpt import DummyGPT
from pretrained_utils import *
from gpt_utils import *
import tiktoken 



# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")
device = torch.device("cpu")
CFG = {
        "vocab_size":50257,
        "context_length":1024,
        "embed_size":768,
        "n_heads":12,
        "n_layers":12,
        "dropout_rate":0.1,
        "qkv_bias":True
    }


MODEL_CFG = {
    "LR" :0.0004,
    "DECAY": 0.1,
    "NUM_EPOCHS":10,
    "EVAL_FREQ":2,
    "TOP_K":50,
    "TEMPERATURE":1,
}
start_context="I missed my childhood years"
gpt_model = DummyGPT(CFG)


MODEL_DIR = "GPT_2_Weights/124M/"
settings,tf_ckpt_path = load_model_params(MODEL_DIR)
gpt_params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path,settings)
load_gpt_params_into_custom_model(gpt_model,gpt_params)


tokenizer = tiktoken.get_encoding("gpt2")
generate_and_print_sample(gpt_model,tokenizer,device,
                          start_context,MODEL_CFG["TOP_K"],CFG["vocab_size"]-1,MODEL_CFG["TEMPERATURE"])
