from torch.utils.data import Dataset
import glob
import numpy as np
from skimage import io
import os.path
import torch
import cv2
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
class VEATIC(Dataset):
    def __init__(self, 
                 character_dir='./data/frames',
                 csv_path='./data/rating_averaged',
                 split=0.7, 
                 clip_length = 5,
                 downsample = 5,
                 skip_size = 1,
                 mode='train'):

        assert mode in ['train', 'test'], 'ERROR: illegal mode passed, must be train or test'
        
        self.character_dir = character_dir
        self.csv_path = csv_path
        self.clip_length = clip_length
        self.downsample = downsample
        self.skip_size = skip_size
        self.video_num = 0
        self.sample_lenth = ((self.clip_length - 1) * self.downsample)
        self.mode = mode

        
        self.legal_frames = []
        cha_vid_dirs = os.listdir(self.character_dir)
        cha_vid_dirs.sort(key=lambda x: int(x))

        for cha_vid in cha_vid_dirs:
            cha_img_paths = glob.glob(f'{self.character_dir}/{cha_vid}/*')
            cha_img_paths.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
            
            length = len(cha_img_paths)

            if self.mode == 'train':
                self.legal_frames.extend(cha_img_paths[self.sample_lenth: int(length * split): self.skip_size])
            else:
                self.legal_frames.extend(cha_img_paths[int(length * split) + self.sample_lenth: : self.skip_size])


    def __len__(self):
        return len(self.legal_frames)

    def __getitem__(self, idx):
        cha_imgs, emotions = [], []
        n_frame = int(self.legal_frames[idx].split('/')[-1].split('.')[0])
        vid = int(self.legal_frames[idx].split('/')[-2])
        arousal_paths = os.path.join(self.csv_path, f'{vid}_arousal.csv')
        valence_paths = os.path.join(self.csv_path, f'{vid}_valence.csv')
        legal_valence = np.loadtxt(valence_paths, delimiter=',')[:,1]
        legal_arousal = np.loadtxt(arousal_paths, delimiter=',')[:,1]
        
        sample_list = [i for i in range(n_frame - self.sample_lenth, n_frame + 1, self.downsample)]
        
        aug_img = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                  transforms.RandomVerticalFlip(p=0.5),
                                  transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
                                  ])
        for i in sample_list:
            cha_image = Image.open(glob.glob(f'{self.character_dir}/{vid}/{i}.jpg')[0])
            val = legal_valence[i]
            aro = legal_arousal[i]
            
            if self.mode == 'train':
                s = np.random.randint(0, 100)
                setup_seed(s)
                cha_image = aug_img(cha_image)

            cha_image =  np.asarray(cha_image) / 128.0 - 1.0

            cha_image = cv2.resize(cha_image, (640, 480)) 

            cha_imgs.append(cha_image)

            emotions.append([val, aro])

        return torch.tensor(np.array(cha_imgs)).float().permute(0, 3, 1, 2), \
                torch.tensor(np.array(emotions[-1])).float().reshape(2)

               

if __name__ == "__main__":
    a = VEATIC(split=0.7, mode='train')

    print(len(a))
