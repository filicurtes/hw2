from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys

import re  ##aggiunte
from collections import defaultdict  ##aggiunte

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        try:
            self.parent = root.split("/")[-2] + "/"
            #print(self.parent)
        except IndexError:
            self.parent = ""

        file_name = self.parent + self.split + ".txt"
        #print(file_name)
        self.files = []
        self.labels = defaultdict(int)

        regex = re.compile("background", re.IGNORECASE) # Don't get those guys
        

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

       #change so that I have an integer for each new label 

        with open(file_name) as f:
            counter = 0
            for line in f:
                if not regex.match(line):
                    label = line.split("/")[0]
                    #print(label)
                    #print(root + "/" + line.rstrip())
                    self.files.append(root + "/" + line.rstrip())
                    if label.lower() not in self.labels.keys():
                        self.labels[label.lower()] = counter
                        counter += 1

        self.length = len(self.files)


    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        # Provide a way to access image and label via index
        # Image should be a PIL Image
        # label can be int

        if self.parent != "":
          label_name = self.files[index].split("/")[2].lower()
        else:
          label_name = self.files[index].split("/")[1].lower()
        
        #print(label_name)
        image, label = (pil_loader(self.files[index]), self.labels[label_name])

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label


    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = self.length # Provide a way to get the length (number of elements) of the dataset
        return length

    def get_labels(self):  ##aggiunto da davide
        return self.labels