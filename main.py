import os

from bottle import route, request, post, run, template
from io import BytesIO
import json
import base64
import time
from datetime import datetime
from PIL import Image
from transformers import BridgeTowerForImageAndTextRetrieval, BridgeTowerProcessor


model_id = "BridgeTower/bridgetower-large-itm-mlm-gaudi"
processor = BridgeTowerProcessor.from_pretrained(model_id)
model = BridgeTowerForImageAndTextRetrieval.from_pretrained(model_id)
LOG_DIR = 'LOG_DIR'
LOG_FILE = LOG_DIR + '/bridgetower.log'

@route('/bridgetower', method='POST')
@route('/bridgetower', method='GET')
def bridgetower():
    start = time.time()
    date_time = datetime.fromtimestamp(start)
    # step 1: parse parameters
    data = json.loads(request.body.read())
    img_pil = Image.open(BytesIO(base64.b64decode(data['image'])))
    texts = data['texts'].split(",")
    scores = {}

    # step 2: run bridge tower
    for text in texts:
        encoding = processor(img_pil, text, return_tensors="pt")
        outputs = model(**encoding)
        scores[text] = "{:.2f}".format(outputs.logits[0, 1].item())
    duration = time.time() - start

    # log into files
    os.makedirs(LOG_DIR, exist_ok=True)
    img_file = f'{LOG_DIR}/{start}.png'
    img_pil.save(img_file)
    with open(LOG_FILE, 'a') as f:
        f.writelines(f"{date_time}\t{data['texts']}\t{img_file}\t{duration}\n")

    # step 3: return
    scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    return scores


run(host='0.0.0.0', port=8080)
