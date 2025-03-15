"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import numpy as np

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-shakespeare-char1' #'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cpu'#'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)

if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)


dataset = 'shakespeare_char'
data_dir = os.path.join('data', dataset)
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 64 #1024
def get_batch_for_test(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    elif split == 'val':
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'test.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# Encode the model's prompt input
#p = encode(prompt)

# 220 is the encoding for a " " character
#pad_len = min(1024 - p.shape[-1], 1024)
#pad = torch.Tensor([220] * pad_len).to(int)
#p = torch.concat([pad.to(int), p[0]]).to(device).to(int)


# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
#x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

#from train_1 import get_batch

X, Y = get_batch_for_test('test')


#x = X[1,:].view(1,64) #torch.load('idx_cond.pt')
x = X#[1,:].view(1,64) #torch.load('idx_cond.pt')

m = torch.export.export(model, (x,)).module()

from torch.ao.quantization.quantize_pt2e import (
  prepare_pt2e,
  convert_pt2e,
)

from torch.ao.quantization.quantizer.xnnpack_quantizer import (
  XNNPACKQuantizer,
  get_symmetric_quantization_config,
)

quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())

m_prep = prepare_pt2e(model=m, quantizer=quantizer)

for iter in range(10):
    X, Y = get_batch_for_test('test')
    m_prep(X)

# calibrate
#for i in range(97,123):
#    start_ids = encode(chr(i))
#    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
#    m_prep(x)

m_quant = convert_pt2e(m_prep)


def generate(Model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= 64 else idx[:, -64:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = Model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        
        import torch.nn.functional as F
        
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def calc_loss(Model, X, Y, temperature=1.0, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    #for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
    #idx_cond = idx if idx.size(1) <= 64 else idx[:, -64:]
    # forward the model to get the logits for the index in the sequence
    logits, loss = Model(X,Y)
    # pluck the logits at the final step and scale by desired temperature
    #logits = logits[:, -1, :] / temperature
    # optionally crop the logits to only the top k options
    #if top_k is not None:
    #    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    #    logits[logits < v[:, [-1]]] = -float('Inf')
    # apply softmax to convert logits to (normalized) probabilities
    
    #import torch.nn.functional as F
    
    #probs = F.softmax(logits, dim=-1)
    # sample from the distribution
    #idx_next = torch.multinomial(probs, num_samples=1)
    # append sampled index to the running sequence and continue
    #idx = torch.cat((idx, idx_next), dim=1)

    return loss

#torch.ops.quantized_decomposed.dequantize_per_tensor.default
eval_iters = 200 #200
from torch.nn import functional as F

@torch.no_grad()
def estimate_loss():
    out = {}
    #model.eval()
    #for split in ['train', 'val']:
    split = 'test'
    losses = torch.zeros(eval_iters)
    losses_q = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch_for_test(split)
        with ctx:
            #logits, loss = model(X, Y)
            logits, _ = model(X)
            logits_q, _ = m_quant(X)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y[:,-1], ignore_index=-1)
            loss_q = F.cross_entropy(logits_q.view(-1, logits.size(-1)), Y[:,-1], ignore_index=-1)
        losses[k] = loss.item()
        losses_q[k] = loss_q.item()
    out = [losses.mean(), losses_q.mean()]
    #model.train()
    return out

# run generation
with torch.no_grad():
    with ctx:
        for k in range(10):
            
            out_loss = estimate_loss()
            print('loss = ', out_loss[0], 'loss_quantized = ', out_loss[1])
            #X, Y = get_batch('test')
            
            #y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            #y = generate(m_quant, x, max_new_tokens, temperature=temperature, top_k=top_k)
            #print(decode(y[0].tolist()),end="")
            #print(decode(y1[0].tolist()),end="")
            #print('---------------')


with torch.no_grad():
    with ctx:
        for k in range(10):
            
            #out_loss = estimate_loss()
            #print('loss = ', out_loss[0], 'loss_quantized = ', out_loss[1])
            #X, Y = get_batch('test')
            
            #y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            y = generate(m_quant, x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()),end="")
            #print(decode(y1[0].tolist()),end="")
            print('---------------')