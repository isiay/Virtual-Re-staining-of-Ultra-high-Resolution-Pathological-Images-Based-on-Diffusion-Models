import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transform=None, classes= "train" , mode='train'):
        self.transform = transform
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/y1' % mode) + '/*.*'))
        x = 0
        # self.files_B = sorted(glob.glob(os.path.join(root, '%s/y1' % mode) + '/*.*'))
        # self.files_A = sorted(glob.glob(os.path.join(root, 'test/x') + '/*.*'))
        # self.files_B = sorted(glob.glob(os.path.join(root, 'test/x') + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        A_name = self.files_A[index % len(self.files_A)].split("/")[-1].split('.')[0]

        # if self.unaligned:
        #     item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        #     B_name = self.files_B[random.randint(0, len(self.files_B) - 1)].split("/")[-1].split('.')[0]
        # else:
        #     item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
        #     B_name = self.files_B[index % len(self.files_B)].split("/")[-1].split('.')[0]

        return item_A, A_name

    def __len__(self):
        return len(self.files_A)