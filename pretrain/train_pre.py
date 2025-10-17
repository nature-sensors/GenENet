import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from models.modules import Encoder, Decoder, Patches, Patches_3d
from models.patchencoder import PatchEncoder
from models.MAE_loss import mae_loss
from utils.scheduler import GradualWarmupScheduler
from utils.train_monitor import trainmonitor
from params import GNETParams
from utils.data_aug import DataAugmentation
import datetime
from utils.dataset import CustomDataset, GetTensorDataset

import torchvision
from tensorboardX import SummaryWriter

from tensorboard import program
import subprocess
import webbrowser
import os
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir=f'logs/{now}'

#writer = SummaryWriter('runs/experiment2')
writer = SummaryWriter(log_dir=log_dir)



torch.manual_seed(0)
global_step=0

def train(epoch, model, train_dataloader, testloader):
    global optimizer,device, global_step
    model.train()
    for step, data in enumerate(train_dataloader):
        inputs, labels = data
        inputs=inputs.to(device)
        optimizer.zero_grad()
        loss = model(inputs)
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        if step%20==0:
            print("epoch: ",epoch, "loss: ", float(loss), "lr", optimizer.param_groups[0]['lr'])
    writer.add_scalar('training_loss', loss, global_step=epoch)
      

def test(epoch,model, test_dataloader):
    global device, params, global_step
    model.eval()
    with torch.no_grad():
        for step, data in enumerate(test_dataloader):
            inputs, labels = data
            inputs=inputs.to(device)
            
        (original, masked, reconstructed)=trainmonitor(model, device, epoch,inputs)
        image_set=torch.stack([original, masked, reconstructed])
        image_set = image_set.permute(0, 3, 1, 2)
        grid = torchvision.utils.make_grid(image_set, nrow=3, normalize=True, scale_each=True)
        writer.add_images('Images', grid.unsqueeze(0), global_step=epoch)

          
def main():
    subprocess.Popen(['tensorboard', '--logdir', log_dir])
    webbrowser.open_new_tab('http://localhost:6006/')

    global device, optimizer, scheduler, params
    params = GNETParams()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_aug=DataAugmentation(params)
              
    mae_model = mae_loss(
        device=device,
        patch_layer=Patches(params),
        patch_layer2=Patches_3d(params),
        patch_encoder=PatchEncoder(params, device),
        encoder=Encoder(params, device),
        decoder=Decoder(params, device),
    )
    model = mae_model
    
    
    tensor_dataset_loader = GetTensorDataset(params)
    data_tensor, label_array= tensor_dataset_loader.get_dataset()
    
    transformations = {
        'train': data_aug.get_train_augmentation_model(),
        'test': data_aug.get_test_augmentation_model()
    }

    train_dataset = CustomDataset((data_tensor, label_array), transform=transformations, train=True)
    test_dataset = CustomDataset((data_tensor, label_array), transform=transformations, train=False)


    train_size = int(0.8 * len(train_dataset))
    test_size = len(test_dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(train_dataset, [train_size, test_size])

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=params.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=params.batch_size, shuffle=True, drop_last=True)
        

                    
            
    warmup_epoch_percentage = 0.15
    total_steps = int(train_size / params.batch_size) * params.num_epoch
    warmup_steps = int(total_steps * warmup_epoch_percentage)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate/10, weight_decay=params.weight_decay)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=0, last_epoch=-1)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=warmup_steps, after_scheduler=cosine_scheduler)

    
    for epoch in range(params.num_epoch):
        train(epoch, model, train_loader,test_loader)
        test(epoch, model, test_loader)


if __name__ == "__main__":
    main()