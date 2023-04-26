##EXAMPLE 1
#### The bare BridgeTower Model transformer outputting BridgeTowerModelOutput object without any specific head on top.",

from transformers import BridgeTowerProcessor, BridgeTowerModel
from PIL import Image
import requests
import torch
import intel_extension_for_pytorch as ipex
import time


wait = 1
warmup = 1
active = 3

def trace_handler(p):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    p.export_chrome_trace("./profile/" + timestr + ".json")
    p.export_stacks("./profile/"+ timestr + ".csv")
    output = p.key_averages().table(sort_by="cpu_time_total", row_limit=20)
    #output = p.key_averages().table(sort_by="cpu_time_total")
    print(output)

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], schedule=torch.profiler.schedule(
    wait=wait,
    warmup=warmup,
    active=active),
    on_trace_ready=trace_handler,
    profile_memory=False,
    with_stack = True,
    with_flops = False,
    with_modules = True,
    record_shapes=True
    ) as prof:

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        text = "hello"
        text="An airplane or aeroplane, informally plane, is is propelled forward by thrust from a jet engine, propeller, or rocket engine. Airplanes come in a variety of sizes, shapes, and wing configurations. The broad spectrum of uses for airplanes includes recreation, transportation of goods and people, military, and research. Worldwide, commercial aviation transports more than four billion passengers annually on transports more than 200 billion kilometers of cargo annually, which is less than 1% of the world's cargo movement. Most airplanes are flown by a pilot on board the aircraft, but some are designed to be remotely or computer-controlled such as drones. The Wright brothers invented and flew the first airplane in 1903, recognized as the first sustained and controlled heavier-than-air powered flight. Aircraft have three main variants. The radial engine is a reciprocating type internal combustion engine configuration in which the cylinders radiate outward from a central crankcase like the spokes of a wheel and was commonly used for aircraft engines before gas turbine engines became predominant. An inline engine is a reciprocating engine with banks of cylinders, one behind another, rather than rows of cylinders, with each bank having any number of cylinders, but rarely more than six, and may be water-cooled. A flat engine is an internal combustion engine with horizontally-opposed cylinders. An airplane or aeroplane, informally plane, is is propelled forward by thrust from a jet engine, propeller, or rocket engine. Airplanes come in a variety of sizes, shapes, and wing configurations. The broad spectrum of uses for airplanes includes recreation, transportation of goods and people, military, and research. Worldwide, commercial aviation transports more than four billion passengers annually on transports more than 200 billion kilometers of cargo annually, which is less than 1% of the world's cargo movement. Most airplanes are flown by a pilot on board the aircraft, but some are designed to be remotely or computer-controlled such as drones. The Wright brothers invented and flew the first airplane in 1903, recognized as the first sustained and controlled heavier-than-air powered flight. Aircraft have three main variants."
        processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base")
        #model = BridgeTowerModel.from_pretrained("BridgeTower/bridgetower-base")
        model = BridgeTowerModel.from_pretrained("BridgeTower/bridgetower-base", torchscript=True)
        inputs = processor(image, text, return_tensors="pt")
        #createe a dict from the  'transformers.tokenization_utils_base.BatchEncoding' dictionary 
        model = ipex.optimize(model)
        model = torch.jit.trace(model,example_kwarg_inputs=dict(inputs), check_trace=False, strict=False)
        model = torch.jit.freeze(model)
        #print(model)
        for iter in range(121):
            with torch.no_grad():
                text="An airplane or aeroplane, informally plane, is is propelled forward by thrust from a jet engine, propeller, or rocket engine. Airplanes come in a variety of sizes, shapes, and wing configurations. The broad spectrum of uses for airplanes includes recreation, transportation of goods and people, military, and research. Worldwide, commercial aviation transports more than four billion passengers annually on transports more than 200 billion kilometers of cargo annually, which is less than 1% of the world's cargo movement. Most airplanes are flown by a pilot on board the aircraft, but some are designed to be remotely or computer-controlled such as drones. The Wright brothers invented and flew the first airplane in 1903, recognized as the first sustained and controlled heavier-than-air powered flight. Aircraft have three main variants. The radial engine is a reciprocating type internal combustion engine configuration in which the cylinders radiate outward from a central crankcase like the spokes of a wheel and was commonly used for aircraft engines before gas turbine engines became predominant. An inline engine is a reciprocating engine with banks of cylinders, one behind another, rather than rows of cylinders, with each bank having any number of cylinders, but rarely more than six, and may be water-cooled. A flat engine is an internal combustion engine with horizontally-opposed cylinders. An airplane or aeroplane, informally plane, is is propelled forward by thrust from a jet engine, propeller, or rocket engine. Airplanes come in a variety of sizes, shapes, and wing configurations. The broad spectrum of uses for airplanes includes recreation, transportation of goods and people, military, and research. Worldwide, commercial aviation transports more than four billion passengers annually on transports more than 200 billion kilometers of cargo annually, which is less than 1% of the world's cargo movement. Most airplanes are flown by a pilot on board the aircraft, but some are designed to be remotely or computer-controlled such as drones. The Wright brothers invented and flew the first airplane in 1903, recognized as the first sustained and controlled heavier-than-air powered flight. Aircraft have three main variants."
                inputs = processor(image, text, return_tensors="pt")
                outputs = model(**inputs)
                prof.step()
            #print(outputs.keys())
        #'text_features', 'image_features', 'pooler_output'
        print(outputs[0].shape)

# ##EXAMPLE 2
from transformers import BridgeTowerProcessor, BridgeTowerForMaskedLM
from PIL import Image
import requests
# #### BridgeTower Model with a language modeling head on top as done during pretraining.
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], schedule=torch.profiler.schedule(
    wait=wait,
    warmup=warmup,
    active=active),
    on_trace_ready=trace_handler,
    profile_memory=False,
    with_stack = True,
    with_flops = False,
    with_modules = True,
    record_shapes=True
    ) as prof:
        url = "http://images.cocodataset.org/val2017/000000360943.jpg"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        text = "a <mask> looking out of the window"
        processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
        model = BridgeTowerForMaskedLM.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
        model = ipex.optimize(model)
        encoding = processor(image, text, return_tensors="pt")
        #createe a dict from the  'transformers.tokenization_utils_base.BatchEncoding' dictionary 
        model = torch.jit.trace(model,example_kwarg_inputs=dict(encoding), check_trace=False, strict=False)
        model = torch.jit.freeze(model)
        #print(model)
        for iter in range(121):
            with torch.no_grad():
                # prepare inputs
                encoding = processor(image, text, return_tensors="pt")
                # forward pass
                outputs = model(**encoding)
                #results = processor.decode(outputs.logits.argmax(dim=-1).squeeze(0).tolist())
                results = processor.decode(outputs['logits'].argmax(dim=-1).squeeze(0).tolist()) #when using jit trace model
                prof.step()
                #print(results)



# ##EXAMPLE 3
# #### BridgeTower Model transformer with a classifier head on top (a linear layer on top of the final hidden state of the [CLS] token) for image-to-text matching.

from transformers import BridgeTowerProcessor, BridgeTowerForImageAndTextRetrieval
import requests
from PIL import Image
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], schedule=torch.profiler.schedule(
    wait=wait,
    warmup=warmup,
    active=active),
    on_trace_ready=trace_handler,
    profile_memory=False,
    with_stack = True,
    with_flops = False,
    with_modules = True,
    record_shapes=True
    ) as prof:
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]
        processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
        model = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
        model = ipex.optimize(model)
        encoding = processor(image, texts[0], return_tensors="pt")
        model = torch.jit.trace(model,example_kwarg_inputs=dict(encoding), check_trace=False, strict=False)
        model = torch.jit.freeze(model)

        #print(model)
        # forward pass
        scores = dict()
        for iter in range(121):
            with torch.no_grad():
                for text in texts:
                # prepare inputs
                    encoding = processor(image, text, return_tensors="pt")
                    outputs = model(**encoding)
                    #scores[text] = outputs.logits[0, 1].item()
                    scores[text] = outputs['logits'][0, 1].item() #when using jit trace model
                    prof.step()
                    #print(scores)

##EXAMPLE 4
#BridgeTower Model with a image-text contrastive head on top computing image-text contrastive loss.
from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning
import requests
from PIL import Image
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], schedule=torch.profiler.schedule(
    wait=wait,
    warmup=warmup,
    active=active),
    on_trace_ready=trace_handler,
    profile_memory=False,
    with_stack = True,
    with_flops = False,
    with_modules = True,
    record_shapes=True
    ) as prof:
        image_urls = [ "https://farm4.staticflickr.com/3395/3428278415_81c3e27f15_z.jpg", "http://images.cocodataset.org/val2017/000000039769.jpg",]
        #texts = ["two dogs in a car", "two cats sleeping on a couch"]
        texts = ["two dogs in a car", "two cats sleeping on a couch"]
        images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]
        processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")

        model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
        model.config.return_dict = False
        model = ipex.optimize(model)
        inputs = processor(images, texts, padding=True, return_tensors="pt")
        #createe a dict from the  'transformers.tokenization_utils_base.BatchEncoding' dictionary 
        input_dict ={}
        input_dict['input_ids'] = inputs['input_ids']
        input_dict['attention_mask'] = inputs['attention_mask']
        input_dict['pixel_values'] = inputs['pixel_values']
        input_dict['pixel_mask'] = inputs['pixel_mask']
        #input_dict['return_dict'] = False
        #input_dict['return_loss'] = True
        model = torch.jit.trace(model,example_kwarg_inputs=input_dict, check_trace=False, strict=False)
        #model.config.use_return_dict=False
        model = torch.jit.freeze(model)
        for iter in range(4):
            with torch.no_grad():
                texts = ["two dogs in a red car", "two cats sleeping on a large couch"]
                inputs = processor(images, texts, padding=True, return_tensors="pt")
                loss = model(**inputs, return_loss=True).loss
                #loss = model(**inputs, return_dict=False, return_loss=True).loss
                prof.step()
                #inputs = processor(images, texts[::-1], padding=True, return_tensors="pt")
                #loss_swapped = model(**inputs, return_loss=True).loss
                print("Loss", round(loss.item(), 4))
                #Loss 0.0019
                #print("Loss with swapped images", round(loss_swapped.item(), 4))


