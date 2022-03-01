import time

import torch.utils.data
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
import os
import PIL.Image as Image
from torchvision.models import resnet50
from os import listdir
from os.path import isfile, join
import numpy as np
from os import path
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
FastRCNNPredictor()

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        #all_imgs = os.listdir(main_dir)
        self.total_imgs = os.listdir(main_dir)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

def load_from_path(path):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),normalize])
    dataset = CustomDataSet(path,transform)
    subsample_to_5_HZ = list(range(0,len(dataset),6))
    dataset = torch.utils.data.Subset(dataset, subsample_to_5_HZ)
    return torch.stack([image for image in dataset])

def image_from_path(path):
    transform = transforms.Compose([transforms.Resize((224,224)),transforms.PILToTensor()])
    dataset = CustomDataSet(path,transform)
    subsample_to_5_HZ = list(range(0,len(dataset),6))
    dataset = torch.utils.data.Subset(dataset, subsample_to_5_HZ)
    return torch.stack([image for image in dataset])

def batch_data(paths):
    batch_tensors = []
    for path in paths:
        batch_tensors.append(load_from_path(path))
    return batch_tensors

def preprocess_images():
    frame_path= '/datashare/APAS/frames/'
    files = [f for f in listdir(frame_path)]
    home_path = '/home/student/Desktop/tensors/'
    features_extractor = resnet50(pretrained=True)
    for file in files:
        start = time.time()
        print(f"starting on file {file}")
        video = load_from_path(frame_path+file)
        with torch.no_grad():
            video_features = features_extractor(video)
            torch.save(video_features,home_path+f'{file}.pt')
        print(f"saving took {time.time()-start:.3f} seconds")
def save_image_tensors():
    frame_path = '/datashare/APAS/frames/'
    files = [f for f in listdir(frame_path)]
    home_path = '/home/student/Desktop/images_tensors/'
    for file in files:
        start = time.time()
        print(f"starting on file {file}")
        video = image_from_path(frame_path + file)
        torch.save(video, home_path + f'{file}.pt')
        print(f"saving took {time.time() - start:.3f} seconds")
#preprocess_images()
save_image_tensors()