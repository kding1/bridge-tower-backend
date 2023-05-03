import os
os.environ['DNNL_GRAPH_VERBOSE'] = '4'
from transformers import BridgeTowerProcessor, BridgeTowerModel, BridgeTowerForMaskedLM, BridgeTowerForImageAndTextRetrieval, BridgeTowerForContrastiveLearning
#from transformers.src.transformers.models.bridgetower.modeling_bridgetower.BridgeTowerResidualAttention
import requests
from PIL import Image
import torch
from torch.profiler import ProfilerActivity, profile, record_function
#import intel_extension_for_pytorch as ipex
import csv
import time
#from deep_learning_performance_advisor.utils import graph_vis
jit = True
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

#blockname="BridgeTowerResidualAttention_vis_noIPEX"
#blk_input = torch.rand(442,1,1024).to(device)
#fwd_graph = model(blk_input).to(device)

# blockname="BridgeTowerTextLayer_noIPEX"
# hidden_states = torch.rand(1,50,1024).to(device)
# atten_mask = torch.rand([1,1,1,50]).to(device)
# fwd_graph = model(hidden_states = hidden_states, attention_mask= atten_mask)

# blockname="BridgeTowerBertCrossLayer_cross_txt_noIPEX"
# hidden_states = torch.rand(1,50,1024).to(device)
# atten_mask = torch.rand(1,1,1,50).to(device)
# enc_atten_mask = torch.rand(1,1,1,442).to(device)
# enc_hidden_states = torch.rand(1,442,1024).to(device)
#fwd_graph = model(hidden_states = hidden_states, attention_mask= atten_mask,encoder_hidden_states=enc_hidden_states,encoder_attention_mask=enc_atten_mask )

blockname="BridgeTowerTextEmbeddings_noIPEX"
input_ids = torch.randint(high=50000, size=(1,50)).to(device)
#fwd_graph = model(input_ids = input_ids)


model= torch.load(blockname +".pt")
model = model.to(device)
print(model)
fwd_graph = model(input_ids = input_ids)
print(fwd_graph)

if profile:
    wait = 1
    warmup = 20
    active = 200
    # Profile
    def trace_handler(p):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        output = p.key_averages().table(sort_by="cuda_time_total", row_limit=20)
        print(output)
        output = p.key_averages().table(sort_by="cuda_time_total")
        if jit:
            with open("./profile_blocks_gpu/"+ blockname + '_gpu_' + timestr + '_jit.csv', 'w+',newline='') as file:
                file.writelines(output)
            output = p.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total")
            with open("./profile_blocks_gpu/"+ blockname + '_gpu_' + timestr + '_byshape_jit.csv', 'w+',newline='') as file:
                file.writelines(output)
        else:
            with open("./profile_blocks_gpu/"+ blockname + '_gpu_' + timestr + '.csv', 'w+',newline='') as file:
                file.writelines(output)
            output = p.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total")
            with open("./profile_blocks_gpu/"+ blockname + '_gpu_' + timestr + '_byshape.csv', 'w+',newline='') as file:
                file.writelines(output)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
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

        for i in range(250):
            #fwd_graph = model(blk_input).to(device)
            #fwd_graph = model(hidden_states = hidden_states, attention_mask= atten_mask)
            #fwd_graph = model(hidden_states = hidden_states, attention_mask= atten_mask,encoder_hidden_states=enc_hidden_states,encoder_attention_mask=enc_atten_mask )
            fwd_graph = model(input_ids = input_ids)
            prof.step()
            if i == wait + warmup + active - 1:
                break
# #load single block
# model= torch.load("./BridgeTowerResidualAttention_vis.pt")
# print(model)
# blk_input = torch.rand(442,1,1024)
# #fwd_graph = model.graph_for(**encoding)
# fwd_graph = model(blk_input)
# print(fwd_graph.shape)
# if jit:
#     model = torch.jit.trace(model,example_kwarg_inputs=dict(encoding), check_trace=False, strict=False)
#     model = torch.jit.freeze(model)

#graph_vis.draw(fwd_graph).render("BridgeTowerResidualAttention_vis") #you can open the demo.svg in Chrome 