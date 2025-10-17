import torch
import torch.nn as nn
import torch.functional as F
import torch.nn.functional as nF
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as nF

class Encoder(nn.Module):
    def __init__(self, params, device):
        super(Encoder, self).__init__()
        self.params = params
        self.device = device
        self.layer_norm1 = nn.LayerNorm(params.enc_proj_dim, eps=params.layer_norm_eps).to(device)
        self.multi_head_attention = nn.MultiheadAttention(params.enc_proj_dim, params.enc_num_heads, dropout=0.1).to(device)
        self.layer_norm2 = nn.LayerNorm(params.enc_proj_dim, eps=params.layer_norm_eps).to(device)
        self.layer_norm3 = nn.LayerNorm(params.enc_proj_dim, eps=params.layer_norm_eps).to(device)
        

    def forward(self, inputs):
        x = inputs
        #x = x.to(device)
        for _ in range(self.params.enc_num_layers):
            x1 = self.layer_norm1(x)
            attention_output, _ = self.multi_head_attention(x1, x1, x1)
            x2 = x + attention_output
            x3 = self.layer_norm2(x2)
            
            for units in self.params.enc_trans_units:
                x3 = nn.Linear(x3.shape[-1], units).to(self.device)(x3)
                x3 = nF.gelu(x3)
                x3 = nn.Dropout(self.params.dropout_rate)(x3)


            x = x3 + x2
        outputs = self.layer_norm3(x)
        return outputs
    
    

class Decoder(nn.Module):
    def __init__(self, params, device):
        super(Decoder, self).__init__()
        self.params = params
        self.device = device
        self.layer_norm1 = nn.LayerNorm(params.dec_proj_dim, params.layer_norm_eps).to(device)
        self.multi_head_attention = nn.MultiheadAttention(params.dec_proj_dim, params.dec_num_heads, dropout=0.1).to(device)
        self.layer_norm2 = nn.LayerNorm(params.dec_proj_dim, params.layer_norm_eps).to(device)
        self.layer_norm3 = nn.LayerNorm(params.dec_proj_dim, params.layer_norm_eps).to(device)
        self.inputlayer = nn.Linear(params.enc_proj_dim, params.dec_proj_dim).to(device)
        self.flatten = nn.Flatten()
        self.pre_final = nn.Linear(params.num_patches*params.dec_proj_dim, params.img_size * params.img_size * 3).to(device)
        self.sigmoid = nn.Sigmoid()
        self.output_layer = nn.Linear(params.img_size* params.img_size * 3, params.img_size * params.img_size * 3).to(device)

    def forward(self, inputs):
        x = self.inputlayer(inputs)
        #x=x.to(device)
        for _ in range(self.params.dec_num_layers):
            x1 = self.layer_norm1(x)
            attention_output, _ = self.multi_head_attention(x1, x1, x1)
            x2 = x + attention_output
            x3 = self.layer_norm2(x2)
            
            for units in self.params.dec_trans_units:
                x3 = nn.Linear(x3.shape[-1], units).to(self.device)(x3)
                x3 = nF.gelu(x3)
                x3 = nn.Dropout(self.params.dropout_rate)(x3)


            x = x3 + x2
        x = self.layer_norm3(x)
        x = self.flatten(x)
        x = self.pre_final(x)
        outputs = self.output_layer(self.sigmoid(x))
        outputs = outputs.view(-1, self.params.img_size, self.params.img_size, 3)


        return outputs




class Patches_3d(nn.Module):
    def __init__(self, params):
        super(Patches_3d, self).__init__()
        self.params = params

    def forward(self, images):
        images=images.permute(0,3,1,2)
        patches = nF.unfold(
            images, 
            (self.params.patch_size, self.params.patch_size), 
            stride=(self.params.patch_size, self.params.patch_size), 
            padding=0,
        )
        patches=patches.permute(0, 2, 1)
        return patches


class Patches(nn.Module):
    def __init__(self, params):
        super(Patches, self).__init__()
        self.params = params
        
    def forward(self, images):
        images=images
        #images=images.to(device)
        patches = nF.unfold(
            images, 
            (self.params.patch_size, self.params.patch_size), 
            stride=(self.params.patch_size, self.params.patch_size), 
            padding=0,
        )
        patches=patches.permute(0, 2, 1) #(batch_size, num_patches, channels * patch_size * patch_size)
        return patches

    def show_patched_image(self, images, patches):
        idx = np.random.choice(patches.shape[0])
        print("Index selected:", idx)

        plt.figure(figsize=(4, 4))
        plt.imshow(images[idx].cpu().permute(1, 2, 0))
        plt.axis("off")
        plt.show()

        n = int(np.sqrt(patches.shape[1]))
        plt.figure(figsize=(4, 4))
        for i, patch in enumerate(patches[idx]):
            ax = plt.subplot(n, n, i + 1)
            patch_img = patch.view(3, self.params.patch_size, self.params.patch_size).permute(1, 2, 0)
            plt.imshow(patch_img.cpu())
            plt.axis("off")
        plt.show()

        return idx

    def reconstruct_from_patch(self, patch):
        num_patches = patch.shape[0]
        n = int(np.sqrt(num_patches))
        patch = patch.view(num_patches, 3, self.params.patch_size, self.params.patch_size)
        rows = patch.split(n, 0)
        rows = [torch.cat(torch.unbind(x, dim=0), dim=2) for x in rows]
        reconstructed = torch.cat(rows, 1).permute(1,2,0)
        return reconstructed



class DownstreamLSTM(nn.Module):
    def __init__(self, device, patch_layer, patch_encoder, encoder, lstm_hidden_size=128, lstm_layers=1, fc_compress_size=32):
        super(DownstreamLSTM, self).__init__()
        self.device = device
        self.patch_layer = patch_layer
        self.patch_encoder = patch_encoder
        self.encoder = encoder
        self.batchnorm = nn.BatchNorm1d(64).to(device)
        self.fc_compress = nn.Linear(64 * 128, fc_compress_size).to(device)
        self.lstm = nn.LSTM(input_size=fc_compress_size, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True).to(device)
        self.fc = nn.Linear(lstm_hidden_size, 26).to(device)
        self.compress_size = fc_compress_size

    def forward(self, x):
        batch_size, sequence_length, C, H, W = x.size()
        cnn_out = torch.zeros(batch_size, sequence_length, self.compress_size).to(self.device)
        for i in range(sequence_length):
            img = self.patch_layer(x[:, i])
            img = self.patch_encoder(img)
            img = self.encoder(img)
            img = self.batchnorm(img)
            img = img.view(img.size(0), -1)
            img = self.fc_compress(img)
            cnn_out[:, i, :] = img

        lstm_out, _ = self.lstm(cnn_out)
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze()

