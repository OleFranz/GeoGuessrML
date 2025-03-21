from train import NeuralNetwork
from train import Device
from train import BatchSize
from train import ImageCount
from train import ImageWidth
from train import ImageHeight
from train import ColorChannels
from train import LearningRate
from train import MaxLearningRate
from train import NumWorkers
from train import Dropout
from train import Patience
from train import Shuffle
from train import ClassCount
from train import Classes
from train import TrainingDatasetSize
from train import ValidationDatasetSize
from PIL import Image
import numpy as np
import datetime
import torch
import os


CheckpointFile = None


Path = os.path.dirname(__file__).replace("\\", "/")

ModelPath = f"{Path}/models"
CheckpointPath = f"{Path}/checkpoints"
os.makedirs(ModelPath, exist_ok=True)
os.makedirs(CheckpointPath, exist_ok=True)

if CheckpointFile != None and os.path.exists(str(CheckpointFile).replace("\\", "/")) and os.path.isfile(str(CheckpointFile).replace("\\", "/")):
    CheckpointFile = str(CheckpointFile).replace("\\", "/")
else:
    CheckpointFile = None

Model = NeuralNetwork()
BestModel = NeuralNetwork()

TotalParameters = 0
for Parameter in Model.parameters():
    TotalParameters += np.prod(Parameter.size())
TrainableParameters = sum(Parameter.numel() for Parameter in Model.parameters() if Parameter.requires_grad)
NonTrainableParameters = TotalParameters - TrainableParameters
BytesPerParameter = next(Model.parameters()).element_size()
ModelSize = (TotalParameters * BytesPerParameter) / (1024 ** 2)


print("Model properties:")
print(f"> Total parameters: {TotalParameters}")
print(f"> Trainable parameters: {TrainableParameters}")
print(f"> Non-trainable parameters: {NonTrainableParameters}")
print(f"> Predicted model size: {ModelSize:.2f}MB")


if CheckpointFile != None:
    print("Loading Checkpoint...")
    Checkpoint = torch.load(CheckpointFile)

    Model.load_state_dict(Checkpoint["Model#StateDict"])
    BestModel.load_state_dict(Checkpoint["BestModel#StateDict"])

    TrainingEpoch = Checkpoint["Model#Epoch"]
    TrainingLoss = Checkpoint["Model#TrainingLoss"]
    ValidationLoss = Checkpoint["Model#ValidationLoss"]

    BestModelEpoch = Checkpoint["BestModel#Epoch"]
    BestModelTrainingLoss = Checkpoint["BestModel#TrainingLoss"]
    BestModelValidationLoss = Checkpoint["BestModel#ValidationLoss"]
else:
    print("No Checkpoint file given!")
    exit()


print("Saving...")

TrainingTime = "Unknown"
TrainingDate = "Unknown"
TrainingTransform = "Unknown"
ValidationTransform = "Unknown"
MetadataOptimizer = "Unknown"
MetadataCriterion = "Unknown"

MetadataModel = str(Model)
Metadata = (f"Epochs#{TrainingEpoch}",
            f"BatchSize#{BatchSize}",
            f"Classes#{Classes}",
            f"Outputs#{ClassCount}",
            f"ImageCount#{ImageCount}",
            f"ImageWidth#{ImageWidth}",
            f"ImageHeight#{ImageHeight}",
            f"ColorChannels#{ColorChannels}",
            f"LearningRate#{LearningRate}",
            f"MaxLearningRate#{MaxLearningRate}",
            f"NumberOfWorkers#{NumWorkers}",
            f"Dropout#{Dropout}",
            f"Patience#{Patience}",
            f"Shuffle#{Shuffle}",
            f"TrainingTime#{TrainingTime}",
            f"TrainingDate#{TrainingDate}",
            f"TrainingDevice#{Device}",
            f"TrainingOS#{os.name}",
            f"Architecture#{MetadataModel}",
            f"TorchVersion#{torch.__version__}",
            f"NumpyVersion#{np.__version__}",
            f"PILVersion#{Image.__version__}",
            f"TrainingTransform#{TrainingTransform}",
            f"ValidationTransform#{ValidationTransform}",
            f"Optimizer#{MetadataOptimizer}",
            f"LossFunction#{MetadataCriterion}",
            f"TrainingDatasetSize#{TrainingDatasetSize}",
            f"ValidationDatasetSize#{ValidationDatasetSize}",
            f"TrainingLoss#{TrainingLoss}",
            f"ValidationLoss#{ValidationLoss}")
Metadata = {"Metadata": Metadata}
Metadata = {Data: str(Value).encode("ascii") for Data, Value in Metadata.items()}

ScriptModel = torch.jit.script(Model)
torch.jit.save(ScriptModel, os.path.join(ModelPath, f"GeoGuessrAI-Last-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.pt"), _extra_files=Metadata)


MetadataModel = str(BestModel)
Metadata = (f"Epochs#{BestModelEpoch}",
            f"BatchSize#{BatchSize}",
            f"Classes#{Classes}",
            f"Outputs#{ClassCount}",
            f"ImageCount#{ImageCount}",
            f"ImageWidth#{ImageWidth}",
            f"ImageHeight#{ImageHeight}",
            f"ColorChannels#{ColorChannels}",
            f"LearningRate#{LearningRate}",
            f"MaxLearningRate#{MaxLearningRate}",
            f"NumberOfWorkers#{NumWorkers}",
            f"Dropout#{Dropout}",
            f"Patience#{Patience}",
            f"Shuffle#{Shuffle}",
            f"TrainingTime#{TrainingTime}",
            f"TrainingDate#{TrainingDate}",
            f"TrainingDevice#{Device}",
            f"TrainingOS#{os.name}",
            f"Architecture#{MetadataModel}",
            f"TorchVersion#{torch.__version__}",
            f"NumpyVersion#{np.__version__}",
            f"PILVersion#{Image.__version__}",
            f"TrainingTransform#{TrainingTransform}",
            f"ValidationTransform#{ValidationTransform}",
            f"Optimizer#{MetadataOptimizer}",
            f"LossFunction#{MetadataCriterion}",
            f"TrainingDatasetSize#{TrainingDatasetSize}",
            f"ValidationDatasetSize#{ValidationDatasetSize}",
            f"TrainingLoss#{BestModelTrainingLoss}",
            f"ValidationLoss#{BestModelValidationLoss}")
Metadata = {"Metadata": Metadata}
Metadata = {Data: str(Value).encode("ascii") for Data, Value in Metadata.items()}

ScriptBestModel = torch.jit.script(Model)
torch.jit.save(ScriptBestModel, os.path.join(ModelPath, f"GeoGuessrAI-Best-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.pt"), _extra_files=Metadata)

print("Done!")