import sys
import numpy as np
import faiss
from collections import defaultdict
import io
import pyarrow as pa

from copy import copy
from PIL import Image
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning
from tqdm import tqdm

import intel_extension_for_pytorch as ipex
from contextlib import nullcontext
import time

device = torch.device('cpu')
bf16 = False

#model_name = 'pretrained/bridgetower-large-itc-itm-mlm-v0.1'
feature_extractor_path = "pretrained/BridgeTowerImageFeatureExtractorLarge.ckpt"
filepath = '/home/maktukma/projects/bridgetower/flickr30k/f30k_caption_karpathy_test.arrow'

model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
model = ipex.optimize(model,
                    auto_kernel_selection=True,
                    dtype=(torch.bfloat16 if bf16 else torch.float32))
processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")

class FlickrDataset(Dataset):
    def __init__(self, filepath):
        
        with pa.memory_map(filepath, 'r') as source:
            self.table = pa.ipc.open_file(source).read_all()
    
    def __len__(self,):
        return len(self.table)
    
    def __getitem__(self, index):
        image_bytes = io.BytesIO(self.table['image'][index].as_py())
        image_bytes.seek(0)
        img = Image.open(image_bytes).convert("RGB")
        caption = self.table['caption'][index][0].as_py()
        image_id = self.table['image_id'][index].as_py()

        return (image_id, caption, img)

def collate_fn(batch_list):
    image_ids, captions, images = list(zip(*batch_list))
    with torch.no_grad():
        with torch.cpu.amp.autocast() if bf16 else nullcontext():
            batch = processor(images, captions, padding=True, return_tensors="pt").to(device)

    batch['image_ids'] = image_ids
    batch['captions'] = captions

    return batch

model.eval()

dataset = FlickrDataset(filepath)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=True, collate_fn=collate_fn)

extracted_embeddings = {}

start_time=time.time()
for _, batch in enumerate(tqdm(dataloader)):    

    captions = copy(batch['captions'])
    image_ids = copy(batch['image_ids'])

    del batch['captions']
    del batch['image_ids']

    with torch.no_grad():
        with torch.cpu.amp.autocast() if bf16 else nullcontext():
            outputs = model(**batch, output_hidden_states=True)

    batch_size = len(captions)
    for bidx in range(batch_size):
        image_id = image_ids[bidx]
        text = captions[bidx]
        extracted_embeddings[image_id] = {
            'caption': text,
            'text_embed': outputs['text_embeds'][bidx],
            'image_embed': outputs['image_embeds'][bidx],
            'cross_embed': outputs['cross_embeds'][bidx],
        }
print("inference time Flickr: ", time.time()-start_time)

image_embeds = []

keys = list(extracted_embeddings.keys())
for image_id in keys:

    image_embed = extracted_embeddings[image_id]['image_embed'].squeeze(0)
    image_embeds.append(image_embed)


vectors = np.stack(image_embeds, axis=0)
num_vectors, vector_dim  = vectors.shape

print(f'Embeddings size: {vector_dim}')
print(f'Index size: {num_vectors}')

index = faiss.IndexFlatIP(vector_dim)
index.add(vectors)

top_ks = [1,5,10]
recall_res = defaultdict(list)
for image_id in keys:

    #text_embed = extracted_embeddings[image_id]['text_embed']
    text_embed = extracted_embeddings[image_id]['text_embed']

    _, I  = index.search(np.array([text_embed.numpy()]), 10)

    retrieved_keys = [keys[i] for i in I[0]]

    for top_k in top_ks:
        if image_id in retrieved_keys[:top_k]:
            recall_res[top_k].append(1)
        else:
            recall_res[top_k].append(0)

#print(model_name)
for top_k in top_ks:
    print(f'Recall@{top_k}: {100*np.mean(recall_res[top_k])}')

