##EXAMPLE 1
#### The bare BridgeTower Model transformer outputting BridgeTowerModelOutput object without any specific head on top.",

from transformers import BridgeTowerProcessor, BridgeTowerModel
from PIL import Image
import requests
import torch
import intel_extension_for_pytorch as ipex


wait = 1
warmup = 1
active = 5

def trace_handler(p):
    output = p.key_averages().table(sort_by="cpu_time_total", row_limit=20)
    print(output)

# with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], schedule=torch.profiler.schedule(
#     wait=wait,
#     warmup=warmup,
#     active=active),
#     on_trace_ready=trace_handler,
#     profile_memory=False,
#     with_stack = False,
#     with_flops = False,
#     with_modules = True,
#     record_shapes=True
#     ) as prof:

#         url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#         image = Image.open(requests.get(url, stream=True).raw)
#         text = "hello world"
#         processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base")
#         model = BridgeTowerModel.from_pretrained("BridgeTower/bridgetower-base")
#         model = ipex.optimize(model)
#         #print(model)
#         for iter in range(7):
#             inputs = processor(image, text, return_tensors="pt")
#             outputs = model(**inputs)
#             prof.step()
#             print(outputs.keys())
#         #'text_features', 'image_features', 'pooler_output'
#         #print(outputs['text_features'])

# ##EXAMPLE 2
# from transformers import BridgeTowerProcessor, BridgeTowerForMaskedLM
# from PIL import Image
# import requests
# # #### BridgeTower Model with a language modeling head on top as done during pretraining.
# with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], schedule=torch.profiler.schedule(
#     wait=wait,
#     warmup=warmup,
#     active=active),
#     on_trace_ready=trace_handler,
#     profile_memory=False,
#     with_stack = False,
#     with_flops = False,
#     with_modules = True,
#     record_shapes=True
#     ) as prof:
#         url = "http://images.cocodataset.org/val2017/000000360943.jpg"
#         image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
#         text = "a <mask> looking out of the window"
#         processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
#         model = BridgeTowerForMaskedLM.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
#         model = ipex.optimize(model)
#         #print(model)
#         for iter in range(7):
#             # prepare inputs
#             encoding = processor(image, text, return_tensors="pt")
#             # forward pass
#             outputs = model(**encoding)
#             results = processor.decode(outputs.logits.argmax(dim=-1).squeeze(0).tolist())
#             prof.step()
#             print(results)



# ##EXAMPLE 3
# #### BridgeTower Model transformer with a classifier head on top (a linear layer on top of the final hidden state of the [CLS] token) for image-to-text matching.

# from transformers import BridgeTowerProcessor, BridgeTowerForImageAndTextRetrieval
# import requests
# from PIL import Image
# with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], schedule=torch.profiler.schedule(
#     wait=wait,
#     warmup=warmup,
#     active=active),
#     on_trace_ready=trace_handler,
#     profile_memory=False,
#     with_stack = False,
#     with_flops = False,
#     with_modules = True,
#     record_shapes=True
#     ) as prof:
#         url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#         image = Image.open(requests.get(url, stream=True).raw)
#         texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]
#         processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
#         model = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
#         model = ipex.optimize(model)
#         #print(model)
#         # forward pass
#         scores = dict()
#         for iter in range(7):
#             for text in texts:
#             # prepare inputs
#                 encoding = processor(image, text, return_tensors="pt")
#                 outputs = model(**encoding)
#                 scores[text] = outputs.logits[0, 1].item()
#             prof.step()
#             print(scores)

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
    with_stack = False,
    with_flops = False,
    with_modules = True,
    record_shapes=True
    ) as prof:
        image_urls = [ "https://farm4.staticflickr.com/3395/3428278415_81c3e27f15_z.jpg", "http://images.cocodataset.org/val2017/000000039769.jpg",]
        texts = ["two dogs in a car", "two cats sleeping on a couch"]
        images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]
        processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
        model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
        model = ipex.optimize(model)
        for iter in range(2):
            inputs = processor(images, texts, padding=True, return_tensors="pt")
            loss = model(**inputs, return_loss=True).loss
            prof.step()
            #inputs = processor(images, texts[::-1], padding=True, return_tensors="pt")
            #loss_swapped = model(**inputs, return_loss=True).loss
            print("Loss", round(loss.item(), 4))
            #Loss 0.0019
            #print("Loss with swapped images", round(loss_swapped.item(), 4))


