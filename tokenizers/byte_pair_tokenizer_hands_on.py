from collections import defaultdict

class BytePairTokenizer():
    def __init__(self,vocab_size):
        self.vocab_size = vocab_size
        self.vocab =[]
        self.merges ={}

    def train(self,corpus):
        self.vocab = list(set(" ".join(corpus)))
        splits = {word:[c for c in word] for word in corpus.split() }
        while len(self.vocab) < self.vocab_size:
            pair_freqs = self._find_pair_freq(splits)
            if not pair_freqs:
                break 
            best_pair =  max(pair_freqs,key = pair_freqs.get)
            self.vocab.append(best_pair[0]+best_pair[1])
            self.merges[best_pair] = best_pair[0]+best_pair[1]
            splits = self._transform_splits(best_pair,splits)
      
       

    def _find_pair_freq(self,splits):
        pair_freqs = defaultdict(int)
        for _,split in splits.items():
           for i in range(len(split)-1):
               pair_freqs[(split[i],split[i+1])] +=1
        return pair_freqs
    def _transform_splits(self,best_pair,splits):
        new_splits = {}
        for word,split in splits.items():
            for i in range(len(split)-1):
                if best_pair[0] == split[i] and best_pair[1] == split[i+1]:
                    split = split[:i] + ["".join(best_pair)] + split[i+2:]
                    break
            new_splits[word] = split
        return new_splits
    
    def tokenize(self,text):
        split = [character for character in text]
        while True:
            for i in range(len(split)-1):
                pair = (split[i],split[i+1])
                if pair in self.merges:
                    split = split[:i] + [self.merges[pair]] +split[i+2:]
                    break           
            return split
        
    def _make_token_map(self):
        token_map = {}
        self.vocab.append("<|endoftext|>")
        for i in range(len(self.vocab)):
            token_map[self.vocab[i]] = i
        return token_map
    def encode(self,text):
        encoded_text=[]
        token_map = self._make_token_map()
        result = self.tokenize(text)
        for token in result:
            encoded_text.append(token_map[token])
        return encoded_text
    
    def decode(self,encoded_text):
        reversed_map =  {code:token for token,code in self._make_token_map().items()}
        text ="".join([reversed_map[code] for code in encoded_text])
        return text




        


with open("../data/the_verdict.txt","r",encoding="UTF-8") as f:
    corpus = f.read()
text ="""I made a deprecating gesture, which he negatived with a good-humoured shrug.

"Oh, I didn't care a straw when I believed in myself--and now it's an added tie between us!"

He laughed slightly, without bitterness, and pushed one of the deep arm-chairs forward. "There: make yourself comfortable--and here are the cigars you like."

He placed them at my elbow and continued to wander up and down the room, stopping now and then beneath the picture.

"How it happened? I can tell you in five minutes--and it didn't take much longer to happen. . . . I can remember now how surprised and pleased I was when I got Mrs. Stroud's note. Of course, deep down, I had always _felt_ there was no one like him--only I had gone with the stream, echoed the usual platitudes about him, till I half got to think he was a failure, one of the kind that are left behind. By Jove, and he _was_ left behind--because he had come to stay! The rest of us had to let ourselves be swept along or go under, but he was high above the current--on everlasting foundations, as you say.

"Well, I went off to the house in my most egregious mood--rather moved, Lord forgive me, at the pathos of poor Stroud's career of failure being crowned by the glory of my painting him! Of course I meant to do the picture for nothing--I told Mrs. Stroud so when she began to stammer something about her poverty. I remember getting off a prodigious phrase about the honour being _mine_--oh, I was princely, my dear Rickham! I was posing to myself like one of my own sitters."""

byte_tokenizer = BytePairTokenizer(100)
byte_tokenizer.train(corpus)
print(byte_tokenizer.merges)
print(byte_tokenizer.vocab)
encoded_text = byte_tokenizer.encode(text)
print("==============Encoding==============")
print(encoded_text)
print("==============Decoding==============")
print(byte_tokenizer.decode(encoded_text))







