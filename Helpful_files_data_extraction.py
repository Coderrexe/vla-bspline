import numpy as np
import os
from PIL import Image
from tqdm import tqdm
    
DATA_PATH = "C:\\Users\\Lenovo\\Desktop\\112" 
OUTPUT_IMG_DIR = "visual_tracking_images"

if not os.path.exists(OUTPUT_IMG_DIR):
    os.makedirs(OUTPUT_IMG_DIR)

files = sorted([f for f in os.listdir(DATA_PATH) if f.endswith('.npz')])

print(f"processing {len(files)} images for tracking...")
for f_name in tqdm(files):
    data = np.load(os.path.join(DATA_PATH, f_name))
    if 'rgb_static' in data:
        img = Image.fromarray(data['rgb_static'])
        img.save(os.path.join(OUTPUT_IMG_DIR, f"{f_name.replace('.npz', '.jpg')}"))

print(f"\finished, file located at: {OUTPUT_IMG_DIR}")
