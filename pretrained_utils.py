##PRETRAINING 
import json
import tensorflow as tf
import torch.nn as nn
import torch
import os
import numpy as np


def load_model_params(model_dir):
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    with open(os.path.join(model_dir,"hparams.json")) as f:
        settings = json.load(f)

    return settings,tf_ckpt_path



def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_value = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_value

    return params

    
def load_gpt_params_into_custom_model(model,gpt_params):
    model.tok_embed.weight = nn.Parameter(torch.tensor(gpt_params["wte"]))
    model.pos_embed.weight = nn.Parameter(torch.tensor(gpt_params["wpe"]))
    for i in range(len(gpt_params["blocks"])):
        # Query,Key,Value Matrises weight assignment 
        Q_W,K_W,V_W= np.split(gpt_params["blocks"][i]["attn"]["c_attn"]["w"],3,axis = -1)
        model.transformer_block[i].multihead_att.Q.weight =  nn.Parameter(torch.tensor(Q_W.T))
        model.transformer_block[i].multihead_att.K.weight =  nn.Parameter(torch.tensor(K_W.T))
        model.transformer_block[i].multihead_att.V.weight =   nn.Parameter(torch.tensor(V_W.T))
        
        # Query,Key,Value Matrises bias assignment
        Q_B,K_B,V_B= np.split(gpt_params["blocks"][i]["attn"]["c_attn"]["b"],3,axis = -1)
        model.transformer_block[i].multihead_att.Q.bias =  nn.Parameter(torch.tensor(Q_B))
        model.transformer_block[i].multihead_att.K.bias =  nn.Parameter(torch.tensor(K_B))
        model.transformer_block[i].multihead_att.V.bias =  nn.Parameter(torch.tensor(V_B))

        # Out Proj Weight and Bias Assignment
        model.transformer_block[i].multihead_att.out_proj.weight = nn.Parameter(torch.tensor(gpt_params["blocks"][i]["attn"]["c_proj"]["w"].T))
        model.transformer_block[i].multihead_att.out_proj.bias = nn.Parameter(torch.tensor(gpt_params["blocks"][i]["attn"]["c_proj"]["b"]))

        # Feed Forward Weight and Bias Assignment
        ## First Layer
        model.transformer_block[i].ff.seq_layers[0].weight = nn.Parameter(torch.tensor(gpt_params["blocks"][i]["mlp"]["c_fc"]["w"].T))
        model.transformer_block[i].ff.seq_layers[0].bias = nn.Parameter(torch.tensor(gpt_params["blocks"][i]["mlp"]["c_fc"]["b"]))
        ## Last Layer
        model.transformer_block[i].ff.seq_layers[2].weight = nn.Parameter(torch.tensor(gpt_params["blocks"][i]["mlp"]["c_proj"]["w"].T))
        model.transformer_block[i].ff.seq_layers[2].bias = nn.Parameter(torch.tensor(gpt_params["blocks"][i]["mlp"]["c_proj"]["b"]))

        #LayerNorm Scale and Shift Weights
        ##Layer 1
        model.transformer_block[i].layernorm1.scale = nn.Parameter(torch.tensor(gpt_params["blocks"][i]["ln_1"]["g"]))
        model.transformer_block[i].layernorm1.shift = nn.Parameter(torch.tensor(gpt_params["blocks"][i]["ln_1"]["b"]))
        ##Layer 2
        model.transformer_block[i].layernorm2.scale = nn.Parameter(torch.tensor(gpt_params["blocks"][i]["ln_2"]["g"]))
        model.transformer_block[i].layernorm2.shift = nn.Parameter(torch.tensor(gpt_params["blocks"][i]["ln_2"]["b"]))

        # out_norm_layer and out_proj_layer Weights and Biases
        model.out_norm_layer.scale = nn.Parameter(torch.tensor(gpt_params["g"]))
        model.out_norm_layer.shift = nn.Parameter(torch.tensor(gpt_params["b"]))

        model.out_proj_layer.weight = nn.Parameter(torch.tensor(gpt_params["wte"]))

