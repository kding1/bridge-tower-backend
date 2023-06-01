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
from utils import disable_quant_conv_bridge_tower, FlickrDataset, MyCollator
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler, BatchSampler
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig

precision = 'int8' #bf16, fp32, int8

device = torch.device('cpu')

processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")

filepath = '/home/maktukma/projects/bridgetower/flickr30k/f30k_caption_karpathy_test.arrow'
dataset = FlickrDataset(filepath)
calib_samples = 32
sampler = SubsetRandomSampler(list(range(calib_samples)))
collate_fn = MyCollator(processor, return_ids_capt = False)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=True, collate_fn=collate_fn, sampler = sampler)
collate_fn = MyCollator(processor, return_ids_capt = True)
dataloader_eval = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=True, collate_fn=collate_fn, sampler = sampler)

model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
model.config.return_dict = False
if precision == 'bf16' or precision == 'fp32':
    model = ipex.optimize(model,
                        auto_kernel_selection=True,
                        dtype=(torch.bfloat16 if precision == 'bf16' else torch.float32))
elif precision == 'int8':
    recipes = {
        "smooth_quant": True,
        "smooth_quant_args": {
            "alpha": 0.5,
            "folding": True,
        },
    }

    op_type_dict = {
        "add": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},
        "linear": {
            "weight": {
                "dtype": ["int8"],
                "scheme": ["sym"],
                "granularity": ["per_channel"],
                "algorithm": ["minmax"],
            },
            "activation": {
                "dtype": ["uint8"],
                "scheme": ["asym"],
                "granularity": ["per_tensor"],
                "algorithm": ["minmax"],
            }
        },
        'Conv2d': {
            'weight': {
                'dtype': ['fp32']
            },
            'activation': {
                'dtype': ['fp32']
            }
        },
        'flatten': {
            'weight': {
                'dtype': ['fp32']
            },
            'activation': {
                'dtype': ['fp32']
            }
        }
    }
    conf = PostTrainingQuantConfig( backend="ipex",
                                    recipes = recipes,
                                    op_type_dict=op_type_dict,) 

    model = quantization.fit(model=model,
                            conf=conf,
                            calib_dataloader=dataloader)

model.eval()


if True:
    extracted_embeddings = {}

    start_time=time.time()
    for _, batch in enumerate(tqdm(dataloader_eval)):    

        captions = copy(batch['captions'])
        image_ids = copy(batch['image_ids'])

        del batch['captions']
        del batch['image_ids']

        with torch.no_grad():
            with torch.cpu.amp.autocast() if precision == 'bf16' else nullcontext():
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

