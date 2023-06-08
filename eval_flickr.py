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
from accelerate import init_empty_weights
from torch.profiler import ProfilerActivity, profile, record_function


evaluate = False
prof = True
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
dataloader_eval = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=True, collate_fn=collate_fn)

model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
model.config.return_dict = False

inp = next(iter(dataloader))

if precision == 'bf16' or precision == 'fp32':
    model = ipex.optimize(model,
                        auto_kernel_selection=True,
                        dtype=(torch.bfloat16 if precision == 'bf16' else torch.float32))
    # Need tracing if the inference performance will be measured
    with torch.no_grad():
        with torch.cpu.amp.autocast() if precision == 'bf16' else nullcontext():
            model = torch.jit.trace(model,example_kwarg_inputs=dict(inp), check_trace=False, strict=False)
            model = torch.jit.freeze(model)

elif precision == 'int8':
    recipes = {
        "smooth_quant": True,
        "smooth_quant_args": {
            "alpha": 0.5,
            "folding": False,
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
        },
        'matmul': {
            'weight': {
                'dtype': ['fp32']
            },
            'activation': {
                'dtype': ['fp32']
            }
        },
    }
    conf = PostTrainingQuantConfig( backend="ipex",
                                    recipes = recipes,
                                    op_type_dict=op_type_dict,) 

    model = quantization.fit(model=model,
                            conf=conf,
                            calib_dataloader=dataloader)

model.eval()
#print('Smoothquant optimized layers:', model.absorb_to_layer.values())

if evaluate:
    
    extracted_embeddings = {}
    start_time=time.time()
    for _, batch in enumerate(tqdm(dataloader_eval)):    

        captions = copy(batch['captions'])
        image_ids = copy(batch['image_ids'])

        del batch['captions']
        del batch['image_ids']

        with torch.no_grad():
            with torch.cpu.amp.autocast() if precision == 'bf16' else nullcontext():
                outputs = model(**batch)

        batch_size = len(captions)
        for bidx in range(batch_size):
            image_id = image_ids[bidx]
            text = captions[bidx]
            extracted_embeddings[image_id] = {
                'caption': text,
                'text_embed': outputs[1][bidx],
                'image_embed': outputs[2][bidx],
                'cross_embed': outputs[3][bidx],
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

if prof:

    wait = 1
    warmup = 5
    active = 10
    # Profile
    def trace_handler(p):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        output = p.key_averages().table(sort_by="cpu_time_total", row_limit=20)
        print(output)
        output = p.key_averages().table(sort_by="cpu_time_total")

    with profile(activities=[ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active),
            on_trace_ready=trace_handler,
            profile_memory=False, 
            with_stack = False,
            with_flops = False,
            with_modules = True,
            record_shapes=True
            ) as prof:

        for i in range(200):
            
            with torch.no_grad():
                output = model(**inp)
            prof.step()
            if i == wait + warmup + active - 1:
                break
