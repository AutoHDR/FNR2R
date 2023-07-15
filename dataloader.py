import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms

def getTrainDataP2(csvPath=None, imgSize=256, validation_split=0):
    df = pd.read_csv(csvPath)
    face = list(df['face'])
    dataset_size = len(face)
    validation_count = int(validation_split * dataset_size)
    train_count = dataset_size - validation_count
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(imgSize)
    ])
    full_dataset = DatasetLoadP2(face, transform)
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

class DatasetLoadP2(Dataset):
    def __init__(self, face, transform=None):

        self.face = face
        self.transform = transform
        self.dataset_len = len(self.face)

    def __getitem__(self, index):
        face_path = self.face[index]
        face = self.transform(Image.open(face_path))
        return face, face_path

    def __len__(self):
        return self.dataset_len

