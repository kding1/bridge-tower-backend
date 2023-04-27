import os
os.environ['DNNL_GRAPH_VERBOSE'] = '4'

from transformers import BridgeTowerProcessor, BridgeTowerModel, BridgeTowerForMaskedLM, BridgeTowerForImageAndTextRetrieval, BridgeTowerForContrastiveLearning
import requests
from PIL import Image
import torch
from torch.profiler import ProfilerActivity, profile, record_function
import intel_extension_for_pytorch as ipex
import csv
import time

task = 'itc' #, 'itc', 'itm', 'mlm'
jit = True

if task == 'mlm': #masked language
    url = "http://images.cocodataset.org/val2017/000000360943.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
#    text = "a <mask> looking out of the window"
    text = "An <mask> or aeroplane, informally plane, is is propelled forward by thrust from a jet engine, propeller, or rocket engine. Airplanes come in a variety of sizes, shapes, and wing configurations. Planes are very good"
    processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
    model = BridgeTowerForMaskedLM.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
    encoding = processor(image, text, return_tensors="pt")
    model = ipex.optimize(model)
    if jit:
        model = torch.jit.trace(model,example_kwarg_inputs=dict(encoding), check_trace=False, strict=False)
        model = torch.jit.freeze(model)
elif task == 'itc': #image txt matching contrastive
    model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
    image_urls = [ "https://farm4.staticflickr.com/3395/3428278415_81c3e27f15_z.jpg", "http://images.cocodataset.org/val2017/000000039769.jpg",]
    #texts = ["two dogs in a car", "two cats sleeping on a couch"]
    texts = ["An airplane or aeroplane, informally plane, is is propelled forward by thrust from a jet engine, propeller, or rocket engine. Airplanes come in a variety of sizes, shapes, and wing configurations. Planes are very good", "two cats sleeping on a couch"]
    images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]
    processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
    model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
    model = ipex.optimize(model)
    model.config.return_dict = False
    #for profiling only passing one image and one text
    encoding = processor(images[0], texts[0], return_tensors="pt")
    if jit:
        model = torch.jit.trace(model,example_kwarg_inputs=dict(encoding), check_trace=False, strict=False)
        model = torch.jit.freeze(model)        
elif task == 'itm': #image-txt matching
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    #texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]
    texts = ["An airplane or aeroplane, informally plane, is is propelled forward by thrust from a jet engine, propeller, or rocket engine. Airplanes come in a variety of sizes, shapes, and wing configurations. Planes are very good", "A football player scoring a goal"]
    processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
    model = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
    encoding = processor(image, texts[0], return_tensors="pt")
    model = ipex.optimize(model)
    if jit:
        model = torch.jit.trace(model,example_kwarg_inputs=dict(encoding), check_trace=False, strict=False)
        model = torch.jit.freeze(model)

if profile:
    wait = 1
    warmup = 20
    active = 100
    # Profile
    def trace_handler(p):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        output = p.key_averages().table(sort_by="cpu_time_total", row_limit=20)
        print(output)
        output = p.key_averages().table(sort_by="cpu_time_total")
        if jit:
            with open("./profile/"+ task + '_' + timestr + '_jit.csv', 'w+',newline='') as file:
                file.writelines(output)
            output = p.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total")
            with open("./profile/"+ task + '_' + timestr + '_byshape_jit.csv', 'w+',newline='') as file:
                file.writelines(output)
        else:
            with open("./profile/"+ task + '_' + timestr + '.csv', 'w+',newline='') as file:
                file.writelines(output)
            output = p.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total")
            with open("./profile/"+ task + '_' + timestr + '_byshape.csv', 'w+',newline='') as file:
                file.writelines(output)



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
            if task == 'mlm':
                with torch.no_grad():
                    encoding = processor(image, text, return_tensors="pt")
                    outputs = model(**encoding)
                    if jit:
                        results = processor.decode(outputs['logits'].argmax(dim=-1).squeeze(0).tolist()) #when using jit trace model
                    else:
                        results = processor.decode(outputs.logits.argmax(dim=-1).squeeze(0).tolist())
                    prof.step()
                    if i == wait + warmup + active - 1:
                        break
            #NOTE for profiling we are only encoding one of the text (single model infereence) instead of two texts and skipping the scoring
            if task == 'itm':
                #scores = dict()
                with torch.no_grad():
                    #for text in texts[0]:
                    encoding = processor(image, texts[0], return_tensors="pt")
                    outputs = model(**encoding)
                    # if jit:
                    #     scores[text] = outputs['logits'][0, 1].item() #when using jit trace model
                    # else:
                    #     scores[text] = outputs.logits[0, 1].item() 
                    #print(scores)
                    prof.step()
                    if i == wait + warmup + active - 1:
                        break
            #NOTE for profiling we are only encoding one text and one image (single model infereence) and not showing loss like in example
            if task == 'itc':
                encoding = processor(images[0], texts[0], return_tensors="pt")
                with torch.no_grad():
                    output = model(**encoding)
                prof.step()
                if i == wait + warmup + active - 1:
                    break
                #example with two inputs and printing loss:
                #texts = ["two dogs in a red car", "two cats sleeping on a large couch"]
                #inputs = processor(images, texts, padding=True, return_tensors="pt")
                #loss = model(**inputs, return_loss=True).loss
                #inputs = processor(images, texts[::-1], padding=True, return_tensors="pt")
                #loss_swapped = model(**inputs, return_loss=True).loss
                #print("Loss", round(loss.item(), 4))