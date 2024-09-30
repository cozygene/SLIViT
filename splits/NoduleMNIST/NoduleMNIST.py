from medmnist import NoduleMNIST3D
# Load Train Data
train_dataset = NoduleMNIST3D(split="train", download=True)
val_dataset = NoduleMNIST3D(split="val", download=True)
test_dataset = NoduleMNIST3D(split="test", download=True)
