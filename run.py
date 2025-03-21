from torchvision import transforms
import numpy as np
import torch
import cv2


ImagePath = None
ModelPath = None


Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Metadata = {"Metadata": []}
Model = torch.jit.load(ModelPath, _extra_files=Metadata, map_location=Device)
Model.eval()
Metadata = eval(Metadata["Metadata"])
for Item in Metadata:
    Item = str(Item)
    if "Classes" in Item:
        Classes = eval(Item.split("#")[1])
    if "ImageWidth" in Item:
        ImageWidth = int(Item.split("#")[1])
    if "ImageHeight" in Item:
        ImageHeight = int(Item.split("#")[1])


Image = cv2.imread(ImagePath, cv2.IMREAD_COLOR_RGB)
Image = cv2.resize(Image, (ImageWidth, ImageHeight))
Image = np.array(Image, dtype=np.float32) / 255.0
Transforms = transforms.Compose([
    transforms.ToTensor()
])
Image = Transforms(Image).unsqueeze(0).to(Device)

with torch.no_grad():
    Output = np.array(Model(Image)[0].tolist())
    Class = np.argmax(Output)
    ClassName = list(Classes.keys())[list(Classes.values()).index(Class)]
    print(f"\nGuessed: {ClassName}\n")

    Results = []
    for ClassName, Index in Classes.items():
        Score = Output[Index] * 100
        Results.append((ClassName, Score))
    Results.sort(key=lambda x: x[1], reverse=True)
    Results = Results[:10]

    LongestClassName = max(len(ClassName) for ClassName, _ in Results)

    for i, (ClassName, Score) in enumerate(Results):
        ClassName += " " * (LongestClassName - len(ClassName))
        print(f"{ClassName}: {round(Score, 3)}%")
    print()