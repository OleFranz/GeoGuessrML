from torchvision import transforms
import SimpleWindow
import threading
import traceback
import ImageUI
import numpy
import torch
import cv2
import os

WindowName = "GeoGuessrML"
WindowX = 200
WindowY = 200
WindowWidth = 960
WindowHeight = 540

Background = numpy.zeros((WindowHeight, WindowWidth, 3), numpy.uint8)
Background[:] = (28, 28, 28)


SimpleWindow.Initialize(Name=WindowName,
                        Size=(WindowWidth, WindowHeight),
                        Position=(WindowX, WindowY),
                        TitleBarColor=(28, 28, 28),
                        Resizable=False,
                        TopMost=False,
                        Undestroyable=False)
SimpleWindow.Show(WindowName, Background)


LastFrame = Background.copy()
ModelLoaded = False
ImageLoaded = False
OriginalImage = Background.copy()
LastTimeAbleToGuess = False
Results = []


def LoadModel(Path):
    global ModelLoaded
    global LastTimeAbleToGuess
    ModelLoaded = False
    LastTimeAbleToGuess = False
    if Path == "": return
    Path = str(Path).replace('"', "").replace("\\", "/")
    if os.path.exists(Path) and os.path.isfile(Path) and Path.endswith(".pt"):
        try:
            global Model, Device, Classes, ImageWidth, ImageHeight
            Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            Metadata = {"Metadata": []}
            Model = torch.jit.load(Path, _extra_files=Metadata, map_location=Device)
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
            ModelLoaded = True
        except:
            traceback.print_exc()
        ImageUI.SetInput(ID="ModelInput", Input=Path)
    else:
        ImageUI.Popup(Text="Invalid model path!",
                      StartX1=Right * 0.3,
                      StartY1=Bottom,
                      StartX2=Right * 0.7,
                      StartY2=Bottom + 20,
                      EndX1=Right * 0.2,
                      EndY1=Bottom - 50,
                      EndX2=Right * 0.8,
                      EndY2=Bottom - 10,
                      ID="InvalidModelPath")
        ImageUI.SetInput(ID="ModelInput", Input="")


def LoadImage(Path):
    global ImageLoaded
    global LastTimeAbleToGuess
    ImageLoaded = False
    LastTimeAbleToGuess = False
    if Path == "": return
    Path = str(Path).replace('"', "").replace("\\", "/")
    if os.path.exists(Path) and os.path.isfile(Path):
        try:
            global OriginalImage, Image
            Image = cv2.imread(Path, cv2.IMREAD_COLOR_RGB)
            OriginalImage = cv2.cvtColor(Image, cv2.COLOR_RGB2BGR).copy()
            Image = cv2.resize(Image, (ImageWidth, ImageHeight))
            Image = numpy.array(Image, dtype=numpy.float32) / 255.0
            Transforms = transforms.Compose([
                transforms.ToTensor()
            ])
            Image = Transforms(Image).unsqueeze(0).to(Device)
            ImageLoaded = True
        except:
            traceback.print_exc()
        ImageUI.SetInput(ID="ImageInput", Input=Path)
    else:
        ImageUI.Popup(Text="Invalid image path!",
                      StartX1=Right * 0.3,
                      StartY1=Bottom,
                      StartX2=Right * 0.7,
                      StartY2=Bottom + 20,
                      EndX1=Right * 0.2,
                      EndY1=Bottom - 50,
                      EndX2=Right * 0.8,
                      EndY2=Bottom - 10,
                      ID="InvalidImagePath")
        ImageUI.SetInput(ID="ImageInput", Input="")


def Guess():
    try:
        global Results
        Results = []
        with torch.no_grad():
            Output = numpy.array(Model(Image)[0].tolist())
        Class = numpy.argmax(Output)
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
    except:
        traceback.print_exc()


Top = 0
Bottom = WindowHeight - 1
Left = 0
Right = WindowWidth - 1


while True:
    Frame = Background.copy()

    if SimpleWindow.GetMinimized(WindowName):
        SimpleWindow.Show(WindowName, LastFrame)
        continue

    if SimpleWindow.GetOpen(WindowName) != True:
        break

    ImageUI.Input(X1=5,
                  Y1=5,
                  X2=Right - 5,
                  Y2=40,
                  ID="ModelInput",
                  Placeholder="Insert the path to the model here!",
                  OnChange=lambda Path: threading.Thread(target=LoadModel, args=(Path,), daemon=True).start())

    if ModelLoaded:
        ImageUI.Input(X1=5,
                    Y1=45,
                    X2=Right - 5,
                    Y2=80,
                    ID="ImageInput",
                    Placeholder="Insert the path to the image here!",
                    OnChange=lambda Path: threading.Thread(target=LoadImage, args=(Path,), daemon=True).start())


    if ModelLoaded and ImageLoaded and LastTimeAbleToGuess == False:
        LastTimeAbleToGuess = True
        ImageUI.Popup(Text="Guessing location...",
                      StartX1=Right * 0.4,
                      StartY1=Bottom,
                      StartX2=Right * 0.6,
                      StartY2=Bottom + 20,
                      EndX1=Right * 0.3,
                      EndY1=Bottom - 50,
                      EndX2=Right * 0.7,
                      EndY2=Bottom - 10,
                      ID="GuessingLocation",
                      ShowDuration=1)
        threading.Thread(target=Guess, daemon=True).start()


    if Results != []:
        for i, (ClassName, Score) in enumerate(Results):
            ImageUI.Button(Text=f"{ClassName}: {round(Score, 3)}%",
                           X1=5,
                           Y1=95 + 35 * i,
                           X2=200,
                           Y2=125 + 35 * i,
                           ID=f"Button{i}")
                           


    ImageUI.Image(Image=OriginalImage,
                  X1=0,
                  Y1=90,
                  X2=Right,
                  Y2=Bottom,
                  ID="Image",
                  Layer=-1)


    Frame = ImageUI.Update(WindowHWND=SimpleWindow.GetHandle(WindowName), Frame=Frame)
    SimpleWindow.Show(WindowName, Frame)
    LastFrame = Frame