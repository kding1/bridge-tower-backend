##EXAMPLE 1
#### The bare BridgeTower Model transformer outputting BridgeTowerModelOutput object without any specific head on top.",

from transformers import BridgeTowerProcessor, BridgeTowerModel
from PIL import Image
import requests
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "hello world"
processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base")
model = BridgeTowerModel.from_pretrained("BridgeTower/bridgetower-base")
inputs = processor(image, text, return_tensors="pt")
outputs = model(**inputs)
print(outputs.keys())
#'text_features', 'image_features', 'pooler_output'
print(outputs['text_features'])

##EXAMPLE 2
#### BridgeTower Model with a language modeling head on top as done during pretraining.

from transformers import BridgeTowerProcessor, BridgeTowerForMaskedLM
from PIL import Image
import requests
url = "http://images.cocodataset.org/val2017/000000360943.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
text = "a <mask> looking out of the window"
processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
model = BridgeTowerForMaskedLM.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
# prepare inputs
encoding = processor(image, text, return_tensors="pt")
# forward pass
outputs = model(**encoding)
results = processor.decode(outputs.logits.argmax(dim=-1).squeeze(0).tolist())
print(results)

##EXAMPLE 3
#### BridgeTower Model transformer with a classifier head on top (a linear layer on top of the final hidden state of the [CLS] token) for image-to-text matching.

from transformers import BridgeTowerProcessor, BridgeTowerForImageAndTextRetrieval
import requests
from PIL import Image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]
processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
model = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
# forward pass
scores = dict()
for text in texts:
# prepare inputs
    encoding = processor(image, text, return_tensors="pt")
    outputs = model(**encoding)
    scores[text] = outputs.logits[0, 1].item()
print(scores)

##EXAMPLE 4
#BridgeTower Model with a image-text contrastive head on top computing image-text contrastive loss.
from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning
import requests
from PIL import Image
import torch
image_urls = [ "https://farm4.staticflickr.com/3395/3428278415_81c3e27f15_z.jpg", "http://images.cocodataset.org/val2017/000000039769.jpg",]
texts = ["two dogs in a car", "two cats sleeping on a couch"]
images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]
processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
inputs = processor(images, texts, padding=True, return_tensors="pt")
loss = model(**inputs, return_loss=True).loss
inputs = processor(images, texts[::-1], padding=True, return_tensors="pt")
loss_swapped = model(**inputs, return_loss=True).loss
print("Loss", round(loss.item(), 4))
#Loss 0.0019
print("Loss with swapped images", round(loss_swapped.item(), 4))

