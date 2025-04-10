
from imports import *
from dataset_classes import AudioDataset, AudioTestDataset
import PHONEMES, config

ROOT = "/kaggle/input/audio_dataset" 

train_data = AudioDataset(root=ROOT, phonemes=PHONEMES, context=config['context'], partition="train-clean-100")

val_data = AudioDataset(root=ROOT, phonemes=PHONEMES, context=config['context'], partition="dev-clean")

test_data = AudioTestDataset(root=ROOT, context=config['context'], partition="test-clean")


train_loader = torch.utils.data.DataLoader(
    dataset     = train_data,
    num_workers = 4,
    batch_size  = config['batch_size'],
    pin_memory  = True,
    shuffle     = True,
    collate_fn = train_data.collate_fn
)

val_loader = torch.utils.data.DataLoader(
    dataset     = val_data,
    num_workers = 2,
    batch_size  = config['batch_size'],
    pin_memory  = True,
    shuffle     = False
)

test_loader = torch.utils.data.DataLoader(
    dataset     = test_data,
    num_workers = 2,
    batch_size  = config['batch_size'],
    pin_memory  = True,
    shuffle     = False
)


# Testing code to check if the validation data loaders are working
all = []
for i, data in enumerate(val_loader):
    frames, phoneme = data
    all.append(phoneme)
    break