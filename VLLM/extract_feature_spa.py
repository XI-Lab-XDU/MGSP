import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# os.environ['HUGGINGFACE_HUB_CACHE'] = '/home/kpn/.cache/huggingface/hub'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# from models.coat import COAT
# from models.clip import clip

import json
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
from PIL import Image
import open_clip
from tqdm import tqdm
import os.path as osp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

ROOT = "/CUHK-SYSU"
OUTPUT_TEXT_DIR = osp.join(ROOT, "text_spatial_caption")
OUTPUT_IMAGE_DIR = osp.join(ROOT, "image_spatial_caption")

# import matplotlib.pyplot as plt

if __name__ == "__main__":

    available_models = open_clip.list_pretrained()
        
    model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai', force_quick_gelu=True)
    model.to('cuda').eval()
    tokenizer = open_clip.get_tokenizer('RN50')

    with open(osp.join(ROOT, "spa_caption.json"), 'r', encoding='utf-8') as file:
        data = json.load(file)  
    image2test_json = {}
    total_image_features = []
    total_text_features = []
    mask_token = tokenizer.encode("mask")
    with torch.no_grad():
        for img_name, _text in tqdm(data.items(), desc="Processing Images"):
            if (ROOT.endswith("PRW")):
                image_path = osp.join(ROOT, "frames", "{}.jpg".format(img_name))

            elif (ROOT.endswith("CUHK-SYSU")):
                image_path = osp.join(ROOT, "Image/SSM", "{}.jpg".format(img_name))

            elif (ROOT.endswith("G2APS")):
                image_path = osp.join(ROOT, "Image/SSM", "{}.jpg".format(img_name))
            else:
                raise Exception("unknown dataset")
            
            image = preprocess(Image.open(image_path)).unsqueeze(0).to('cuda')
            image_features = model.encode_image(image)
            text = tokenizer(_text).to('cuda')
            text_features, text_local_feature = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            total_image_features.append(image_features)
            total_text_features.append(text_features)

            os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
            os.makedirs(OUTPUT_TEXT_DIR, exist_ok=True)
            torch.save(image_features.cpu(), osp.join(OUTPUT_IMAGE_DIR, f"{img_name}.pt"))
            torch.save(text_features.cpu(), osp.join(OUTPUT_TEXT_DIR, f"{img_name}.pt"))
    
    total_image_features = torch.cat(total_image_features)
    total_text_features = torch.cat(total_text_features)
   
    
    
    
    