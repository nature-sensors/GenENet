import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from sklearn.model_selection import train_test_split
from models.modules import Encoder, Decoder, Patches, Patches_3d, DownstreamLSTM
from models.patchencoder import PatchEncoder
from models.MAE_loss import mae_loss
from utils.scheduler import GradualWarmupScheduler
from utils.data_aug import DataAugmentation
from utils.dataset import GetTensorDataset_Threshold_Sequence
from params import GNETParams

def train():
    params = GNETParams()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_aug = DataAugmentation(params)

    mae_model = mae_loss(
        device=device,
        patch_layer=Patches(params),
        patch_layer2=Patches_3d(params),
        patch_encoder=PatchEncoder(params, device, downstream=True),
        encoder=Encoder(params, device),
        decoder=Decoder(params, device),
    )
    mae_model.load_state_dict(torch.load(params.pretrained_path))
    patch_layer, encoder, _, patch_encoder = mae_model.layer_prop()

    down_model = DownstreamLSTM(device, patch_layer, patch_encoder, encoder, fc_compress_size=32)
    down_model.to(device)

    transformations = {
        'train': data_aug.get_application_test_augmentation_model(),
        'test': data_aug.get_application_test_augmentation_model()
    }

    train_dataset = GetTensorDataset_Threshold_Sequence(params, train=True, sequence_length=7, transform=transformations)
    test_dataset = GetTensorDataset_Threshold_Sequence(params, train=False, sequence_length=7, transform=transformations)

    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, drop_last=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, drop_last=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(down_model.parameters(), lr=0.003 / 10)
    warmup_epoch_percentage = 0.15
    total_steps = int(len(train_loader) * params.num_epoch)
    warmup_steps = int(total_steps * warmup_epoch_percentage)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=0, last_epoch=-1)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=warmup_steps, after_scheduler=cosine_scheduler)

    best_test_acc = 0.0

    for epoch in range(params.num_epoch):
        down_model.train()
        train_loss, train_correct = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = down_model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * inputs.size(0)
            train_correct += torch.sum(preds == labels)

        train_acc = train_correct.double() / len(train_loader.dataset)

        # Evaluation
        down_model.eval()
        test_correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = down_model(inputs)
                _, preds = torch.max(outputs, 1)
                test_correct += torch.sum(preds == labels)

        test_acc = test_correct.double() / len(test_loader.dataset)
        print(f"Epoch {epoch + 1}/{params.num_epoch} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    train()