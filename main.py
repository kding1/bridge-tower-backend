from bottle import route, request, post, run, template
from io import BytesIO
from PIL import Image
import json, base64, os, socket


IMG_ROOT = 'concept_images'

@route('/finetuning', method='POST')
def finetuning():
    print('i am called')
    os.makedirs(IMG_ROOT, exist_ok=True)

    # step 1: parse parameters
    data = json.loads(request.body.read())
    for i, img in enumerate(data['files_to_upload']):
        img_byte = base64.b64decode(img)
        img_pil = Image.open(BytesIO(img_byte))
        img_pil.save(f'{IMG_ROOT}/{i + 1}.png')

    # step 2: launch worker job, need to change aws
    hostname = socket.gethostname()
    os.system(f'/home/ec2-user/aia-diffuser-opt/third_party/diffusers/examples/textual_inversion/training_ddp_kding1.sh {hostname}')

    # step 3: return
    return {"model_name": "kding1/dicoo_model_ddp"}


run(host='0.0.0.0', port=8080)

