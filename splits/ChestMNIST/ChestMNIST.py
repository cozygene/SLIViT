from medmnist import ChestMNIST
train_dataset = ChestMNIST(split="train", download=True)
val_dataset = ChestMNIST(split="val", download=True)
test_dataset = ChestMNIST(split="test", download=True)
