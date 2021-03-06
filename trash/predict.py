import os

import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from pkg_resources import resource_filename
from tqdm import tqdm

config = OmegaConf.load(resource_filename(__name__, "configs/config.yaml"))

model = torch.load(
    f"{config['checkpoints']}/{config['image_size']}_{config['model']}.pt"
)
model.to("cuda")
model.eval()

meta_info = pd.read_csv("data/sample_solution.csv")
test_data = [
    os.path.join(config["data_path"], "test", f"{i}.jpg")
    for i in meta_info.ID_img.values
]

fin_outputs = []
with torch.no_grad():
    for i in tqdm(test_data, total=len(test_data)):
        image = cv2.imread(i)
        image = cv2.resize(image, (config["image_size"], config["image_size"]))
        image = torch.tensor(np.moveaxis(image, -1, 0), dtype=torch.float).unsqueeze(0)
        image = image.to(config["device"])

        outputs = model(image)
        outputs = outputs.squeeze(1)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        fin_outputs.extend(outputs.tolist())


meta_info["class"] = fin_outputs
meta_info.to_csv("submission.csv", index=None)
