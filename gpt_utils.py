import torch

def generate_next_words(model,X_ids,max_new_tokens,context_length,top_k=None,temp=0.0,eos_id = None):
    for _ in range(max_new_tokens):
        cond_ids = X_ids[:,-context_length:]
        with torch.no_grad():
            logits = model(cond_ids)
            logits = logits[:,-1,:]
            if top_k is not None:
                top_logits,_ = torch.topk(logits,top_k)
                logits = torch.where(logits < top_logits[:,-1],torch.tensor(float("-inf")).to(logits.device),logits)
            if temp > 0.0:
                logits /=temp
                probas = torch.softmax(logits,dim = -1)
                idx_next = torch.multinomial(probas,num_samples= 1)
            else:
                idx_next = torch.argmax(logits,dim = -1,keepdim=True)

            if idx_next == eos_id:
                break

            X_ids = torch.cat([X_ids,idx_next],dim=-1)
    return X_ids

def encode_text_token(text,tokenizer):
    X_ids = tokenizer.encode(text,allowed_special = {"<|endoftext|>"})
    X_ids = torch.tensor([X_ids])

    return X_ids

def decode_token_ids(X_ids,tokenizer):
   return tokenizer.decode(X_ids[0].tolist())


def generate_and_print_sample(model, tokenizer, device, start_context,top_k=None,eos_id=None,temp=0.0):
    model.eval()
    context_length = model.pos_embed.weight.shape[0]
    encoded = encode_text_token(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_next_words(
            model=model, X_ids=encoded,
            max_new_tokens=50, context_length=context_length,top_k = top_k,temp=temp,eos_id=eos_id
        )
    decoded_text = decode_token_ids(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss