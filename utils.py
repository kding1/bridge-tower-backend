import torch
from intel_extension_for_pytorch.quantization._quantization_state_utils import SeenQOpInfo, SeenNonQOpInfo, QTensorInfo
from torch.utils.data import Dataset, DataLoader
import pyarrow as pa
import io
from PIL import Image

def disable_quant_conv_bridge_tower(model):

    # Disable conv2d quant in image embeddings
    set_q = model._fqn_to_auto_quant_state_map['bridgetower:vision_model:visual:embeddings'].idx_to_seen_q_op_infos
    set_q[0].input_tensor_infos[0] = QTensorInfo(set_q[0].input_tensor_infos[0].id, torch.float32, torch.float32)
    set_q[0].input_tensor_force_inf_dtype[0] = torch.float32
    set_q[0].output_tensor_infos[0] = QTensorInfo(set_q[0].output_tensor_infos[0].id, torch.float32, torch.float32)
    set_q[0].weight_tensor_infos[0] = QTensorInfo(set_q[0].weight_tensor_infos[0].id, torch.float32, torch.float32)
    model._fqn_to_auto_quant_state_map['bridgetower:vision_model:visual:embeddings'].idx_to_op_weight_convert_info[0][1][0] = False
    model._fqn_to_auto_quant_state_map['bridgetower:vision_model:visual:embeddings'].idx_to_op_convert_info[0][1][0] = False


class FlickrDataset(Dataset):
    def __init__(self, filepath):
        
        with pa.memory_map(filepath, 'r') as source:
            self.table = pa.ipc.open_file(source).read_all()
    
    def __len__(self,):
        return len(self.table)
    
    def __getitem__(self, index):
        image_bytes = io.BytesIO(self.table['image'][index].as_py())
        image_bytes.seek(0)
        img = Image.open(image_bytes).convert("RGB")
        caption = self.table['caption'][index][0].as_py()
        image_id = self.table['image_id'][index].as_py()

        return (image_id, caption, img)
    

class MyCollator(object):
    def __init__(self, processor, return_ids_capt = True):
        self.processor = processor
        self.return_ids_capt = return_ids_capt
    def  __call__(self, batch_list):
        image_ids, captions, images = list(zip(*batch_list))
        with torch.no_grad():
            batch = self.processor(images, captions, padding=True, return_tensors="pt").to('cpu')

        if self.return_ids_capt:
            batch['image_ids'] = image_ids
            batch['captions'] = captions

        return batch