
import re 

class SimpleTokenizer:
    def __init__(self,vocab):
        self.token2id = vocab
        self.id2token = {id:token for token,id in vocab.items()}


    def encode(self,text):
        t_text=re.split(r'([,.:;?!"()\']|--|\s)',text)
        t_text = [word.strip() for word in t_text if word.strip()]
        t_ids = [self.token2id[token] if token in self.token2id else self.token2id["<|unk|>"]for token in t_text]

        return t_ids
    
    def decode(self,ids):
        text = " ".join([self.id2token[id] for id in ids])
        # we need to remove the spaces after punctuations
        text = re.sub(r'\s([,.:;?!"()\'])',r'\1',text)
        return text




with open("./data/the_verdict.txt","r",encoding="UTF-8") as f:
    raw_text = f.read()
print(f"Total number of characters: {len(raw_text)}")

tokenized_text = re.split(r'([,.:;?!"()\']|--|\s)',raw_text)
tokenized_text = [word.strip() for word in tokenized_text if word.strip()]

print(f"How many tokens we have: {len(tokenized_text)}")

tokenized_text = sorted(set(tokenized_text))
tokenized_text.extend(["<|endoftext|>","<|unk|>"])
vocab ={token:id for id,token in enumerate(tokenized_text)}

## Tokenizer Test 
text1 = "Hello,do you like tea?"
text2 = "Turkish people like tea a lot"
text = " <|endoftext|> ".join([text1,text2])
s_tokenizer = SimpleTokenizer(vocab)
t_ids = s_tokenizer.encode(text)
embed_text = s_tokenizer.decode(t_ids)
print(embed_text)
