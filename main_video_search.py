import os
from bottle import route, request, post, run, template
from io import BytesIO
import json
import base64
import time
import tempfile
import cv2
from datetime import datetime
from PIL import Image
from transformers import BridgeTowerForImageAndTextRetrieval, BridgeTowerProcessor


model_id = "BridgeTower/bridgetower-large-itm-mlm"
processor = BridgeTowerProcessor.from_pretrained(model_id)
model = BridgeTowerForImageAndTextRetrieval.from_pretrained(model_id)
LOG_DIR = 'LOG_DIR_VIDEO'
LOG_FILE = LOG_DIR + '/bridgetower_vid.log'


# Process a frame
def process_frame(image, texts):
    scores = {}
    texts = texts.split(",")
    for t in texts:
        encoding = processor(image, t, return_tensors="pt")
        outputs = model(**encoding)
        scores[t] = "{:.2f}".format(outputs.logits[0, 1].item())
        # sort scores in descending order
        scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    return scores


@route('/btvideo', method='POST')
@route('/btvideo', method='GET')
def btvideo():
    start = time.time()
    date_time = datetime.fromtimestamp(start)
    # step 1: parse parameters
    data = json.loads(request.body.read())
    os.makedirs(LOG_DIR, exist_ok=True)
    vid_file = f'{LOG_DIR}/{start}.{data["suffix"]}'
    with open(vid_file, 'wb') as f:
        f.write(base64.b64decode(data['video']))
    text = data['text']
    sample_rate = data["sample_rate"]
    min_score = data["min_score"]

    # step 2: run bridge tower
    video = cv2.VideoCapture(vid_file)
    fps = round(video.get(cv2.CAP_PROP_FPS))
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    length = frames // fps
    print(f"{fps} fps, {frames} frames, {length} seconds")

    frame_count = 0
    clips = []
    clip_images = []
    clip_started = False
    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % (fps * sample_rate) == 0:
            frame = Image.fromarray(frame)
            score = process_frame(frame, text)
            # print(f"{frame_count} {scores}")

            if float(score[text]) > min_score:
                if clip_started:
                    end_time = frame_count / fps
                else:
                    clip_started = True
                    start_time = frame_count / fps
                    end_time = start_time
                    start_score = score[text]
                    buffered = BytesIO()
                    frame.save(buffered, format="PNG")
                    img_b64 = base64.b64encode(buffered.getvalue())
                    clip_images.append(img_b64.decode())
            elif clip_started:
                clip_started = False
                end_time = frame_count / fps
                clips.append((start_score, start_time, end_time))
        frame_count += 1
    duration = time.time() - start

    # log into files
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.writelines(f"{date_time}\t{data['text']}\t{data['sample_rate']}\t{data['min_score']}\t"
                     f"{vid_file}\t{duration}\n")

    # step 3: return
    return {"clip_images": clip_images, "clips": clips}


run(host='0.0.0.0', port=443)
