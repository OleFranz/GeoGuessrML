from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from torch.amp import GradScaler
import torch.optim as optim
import torch.nn as nn
import threading
import random
import torch
import time
import cv2
import os


Path = os.path.dirname(__file__).replace("\\", "/")
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Epochs = 1000
BatchSize = 8
ImageWidth = int(1600 * 0.35)
ImageHeight = int(900 * 0.35)
ColorChannels = 3
LearningRate = 0.001
MaxLearningRate = 0.001
Dropout = 0.1
Shuffle = True
DropLast = True
PinMemory = True


ImagesPerClass = -1
DataPath = f"{Path}/dataset/hub"
if os.path.exists(DataPath):
    DatasetFolder = [Folder for Folder in os.listdir(DataPath) if os.path.isdir(f"{DataPath}/{Folder}") and "dataset" in Folder.lower()][0]
    DataPath += f"/{DatasetFolder}/snapshots"
    SnapshotFolder = [Folder for Folder in os.listdir(DataPath) if os.path.isdir(f"{DataPath}/{Folder}")][0]
    DataPath += f"/{SnapshotFolder}"
    ClassCount = max(len(os.listdir(f"{DataPath}/train")), len(os.listdir(f"{DataPath}/test")))
    Classes = {}
    TrainingFiles = []
    ValidationFiles = []
    for Folder in os.listdir(f"{DataPath}/train"):
        if Folder not in Classes.keys():
            Classes[Folder] = len(Classes.keys())
        i = 0
        for File in os.listdir(f"{DataPath}/train/{Folder}"):
            TrainingFiles.append(f"{DataPath}/train/{Folder}/{File}")
            i += 1
            if ImagesPerClass > 0 and i >= ImagesPerClass:
                break
    for Folder in os.listdir(f"{DataPath}/test"):
        if Folder not in Classes.keys():
            Classes[Folder] = len(Classes.keys())
        i = 0
        for File in os.listdir(f"{DataPath}/test/{Folder}"):
            ValidationFiles.append(f"{DataPath}/test/{Folder}/{File}")
            i += 1
            if ImagesPerClass > 0 and i >= ImagesPerClass:
                break
    ImageCount = len(TrainingFiles) + len(ValidationFiles)
    TrainingDatasetSize = len(TrainingFiles)
    ValidationDatasetSize = len(ValidationFiles)
else:
    print("No dataset found, exiting...")
    exit()


class NeuralNetwork(nn.Module):
    def __init__(Self):
        super(NeuralNetwork, Self).__init__()
        
        Self.Conv2d_1 = nn.Conv2d(ColorChannels, 32, (3, 3), padding=1, bias=False)
        Self.BatchNorm2d_1 = nn.BatchNorm2d(32)
        Self.ReLU_1 = nn.ReLU()
        Self.MaxPool2d_1 = nn.MaxPool2d((2, 2))

        Self.Conv2d_2 = nn.Conv2d(32, 64, (3, 3), padding=1, bias=False)
        Self.BatchNorm2d_2 = nn.BatchNorm2d(64)
        Self.ReLU_2 = nn.ReLU()
        Self.MaxPool2d_2 = nn.MaxPool2d((2, 2))

        Self.Conv2d_3 = nn.Conv2d(64, 128, (3, 3), padding=1, bias=False)
        Self.BatchNorm2d_3 = nn.BatchNorm2d(128)
        Self.ReLU_3 = nn.ReLU()
        Self.MaxPool2d_3 = nn.MaxPool2d((2, 2))

        Self.Conv2d_4 = nn.Conv2d(128, 256, (3, 3), padding=1, bias=False)
        Self.BatchNorm2d_4 = nn.BatchNorm2d(256)
        Self.ReLU_4 = nn.ReLU()
        Self.MaxPool2d_4 = nn.MaxPool2d((2, 2))

        Self.Conv2d_5 = nn.Conv2d(256, 512, (3, 3), padding=1, bias=False)
        Self.BatchNorm2d_5 = nn.BatchNorm2d(512)
        Self.ReLU_5 = nn.ReLU()
        Self.MaxPool2d_5 = nn.MaxPool2d((2, 2))

        Self.Flatten = nn.Flatten()
        Self.Dropout = nn.Dropout(Dropout)
        Self.Linear_1 = nn.Linear(512 * (ImageWidth // 32) * (ImageHeight // 32), 1024, bias=False)
        Self.BatchNorm1d = nn.BatchNorm1d(1024)
        Self.ReLU_4 = nn.ReLU()
        Self.Linear_2 = nn.Linear(1024, ClassCount, bias=False)
        Self.Softmax = nn.Softmax(dim=1)

    def forward(Self, X):
        X = Self.Conv2d_1(X)
        X = Self.BatchNorm2d_1(X)
        X = Self.ReLU_1(X)
        X = Self.MaxPool2d_1(X)

        X = Self.Conv2d_2(X)
        X = Self.BatchNorm2d_2(X)
        X = Self.ReLU_2(X)
        X = Self.MaxPool2d_2(X)

        X = Self.Conv2d_3(X)
        X = Self.BatchNorm2d_3(X)
        X = Self.ReLU_3(X)
        X = Self.MaxPool2d_3(X)

        X = Self.Conv2d_4(X)
        X = Self.BatchNorm2d_4(X)
        X = Self.ReLU_4(X)
        X = Self.MaxPool2d_4(X)

        X = Self.Conv2d_5(X)
        X = Self.BatchNorm2d_5(X)
        X = Self.ReLU_5(X)
        X = Self.MaxPool2d_5(X)

        X = Self.Flatten(X)
        X = Self.Dropout(X)
        X = Self.Linear_1(X)
        X = Self.BatchNorm1d(X)
        X = Self.ReLU_4(X)
        X = Self.Linear_2(X)
        X = Self.Softmax(X)
        return X


TrainingTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop((round(ImageHeight * random.uniform(0.75, 1)), round(ImageWidth * random.uniform(0.75, 1)))),
    transforms.Resize((ImageHeight, ImageWidth))
])

ValidationTransform = transforms.Compose([
    transforms.ToTensor()
])


class CustomDataset(Dataset):
    def __init__(Self, Files, Transform, Shuffle, BatchSize, BatchPreloadCount):
        Self.Cache = {}
        Self.LastUsedIndex = None
        Self.UseFiles = []
        Self.Files = Files
        Self.Transform = Transform
        Self.Shuffle = Shuffle
        Self.BatchSize = BatchSize
        Self.BatchPreloadCount = BatchPreloadCount

    def __len__(Self):
        return len(Self.Files)

    def CacheIndex(Self, Index):
        if Index == 0:
            Self.Cache = {}
            Self.LastUsedIndex = None
        if str(Index) not in Self.Cache:
            Self.Cache[str(Index)] = {}
            Self.Cache[str(Index)]["FullyCached"] = False
        threading.Thread(target=Self.CacheIndexThread, args=(Index,), daemon=True).start()

    def CacheIndexThread(Self, Index):
        if Index == 0:
            if Self.Shuffle:
                Self.UseFiles = random.sample(Self.Files, len(Self.Files))
            else:
                Self.UseFiles = Self.Files

        File = Self.UseFiles[Index]

        Image = cv2.imread(File, cv2.IMREAD_COLOR_RGB)
        Image = cv2.resize(Image, (ImageWidth, ImageHeight))
        Image = Image / 255.0
        Image = Self.Transform(Image)
        Image = torch.as_tensor(Image, dtype=torch.float32, device=Device)

        Class = Classes[os.path.basename(os.path.dirname(File))]
        Label = int(Class)
        Label = torch.as_tensor(Label, device=Device)

        Self.Cache[str(Index)]["Image"] = Image
        Self.Cache[str(Index)]["Label"] = Label
        Self.Cache[str(Index)]["FullyCached"] = True

    def ClearIndex(Self, Index):
        if str(Index) in Self.Cache:
            del Self.Cache[str(Index)]

    def GetIndex(Self, Index):
        if str(Index) not in Self.Cache:
            Self.CacheIndex(Index)
            for i in range(Self.BatchPreloadCount * Self.BatchSize):
                Self.CacheIndex(Index + 1 + i)
        Self.CacheIndex(Index + Self.BatchPreloadCount * Self.BatchSize)
        if Self.LastUsedIndex != None:
            Self.ClearIndex(Self.LastUsedIndex)
        while Self.Cache[str(Index)]["FullyCached"] == False:
            time.sleep(0.0001)
        Self.LastUsedIndex = Index
        return Self.Cache[str(Index)]["Image"], Self.Cache[str(Index)]["Label"]

    def __getitem__(Self, Index):
        return Self.GetIndex(Index)

TrainingDataset = CustomDataset(TrainingFiles, TrainingTransform, Shuffle, BatchSize, 3)
ValidationDataset = CustomDataset(ValidationFiles, ValidationTransform, Shuffle, BatchSize, 3)

TrainingDataloader = DataLoader(TrainingDataset, batch_size=BatchSize, shuffle=False, num_workers=0, pin_memory=False, drop_last=DropLast)
ValidationDataloader = DataLoader(ValidationDataset, batch_size=BatchSize, shuffle=False, num_workers=0, pin_memory=False, drop_last=DropLast)


Model = NeuralNetwork().to(Device)
Model.train()
Scaler = GradScaler(device=str(Device))
Criterion = nn.CrossEntropyLoss()
Optimizer = optim.Adam(Model.parameters(), lr=LearningRate)
Scheduler = lr_scheduler.OneCycleLR(Optimizer, max_lr=MaxLearningRate, steps_per_epoch=len(TrainingDataloader), epochs=Epochs)

for Epoch in range(Epochs):
    EpochStart = time.perf_counter()
    for i, (Images, Labels) in enumerate(TrainingDataloader, 0):
        Images = Images.to(Device)
        Labels = Labels.to(Device)

        Optimizer.zero_grad()

        Outputs = Model(Images)
        Loss = Criterion(Outputs, Labels)

        Scaler.scale(Loss).backward()
        Scaler.step(Optimizer)
        Scaler.update()
        Scheduler.step()

        print(f"Epoch {Epoch + 1}/{Epochs}, Step {i + 1}/{len(TrainingDataloader)}: Loss: {Loss.item()} Time left: {time.strftime('%H:%M:%S', time.gmtime((time.perf_counter() - EpochStart) / (i + 1) * len(TrainingDataloader)))}")