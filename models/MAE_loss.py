
"""
This code is implemented from MAE (Masked Autoencoders Are Scalable Vision Learners).
For the original MAE code, please refer to "https://github.com/facebookresearch/mae"

"""

import torch
import torch.nn as nn
from models.modules import Encoder, Decoder, Patches, Patches_3d
from models.patchencoder import PatchEncoder


class mae_loss(nn.Module):
    def __init__(self,device, patch_layer, patch_layer2, patch_encoder, encoder, decoder):
        super().__init__()
        
        self.loss_fn = nn.MSELoss().to(device)
        self.patch_layer = patch_layer
        self.patch_layer2 = patch_layer2
        self.patch_encoder = patch_encoder
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def layer_prop(self):
       return self.patch_layer, self.encoder, self.decoder, self.patch_encoder

    def calculate_loss(self, images, test=False):
       
        patches = self.patch_layer(images)
        unmasked_embeddings, masked_embeddings, unmasked_positions, mask_indices, unmask_indices = self.patch_encoder(patches)

        encoder_outputs = self.encoder(unmasked_embeddings)

        encoder_outputs = encoder_outputs + unmasked_positions
        decoder_inputs = torch.cat([encoder_outputs, masked_embeddings], dim=1)

        decoder_outputs = self.decoder(decoder_inputs)
        decoder_patches = self.patch_layer2(decoder_outputs)
        

        loss_patch = patches.gather(dim=1, index=mask_indices.unsqueeze(-1).expand(-1, -1, patches.size(-1)))
        loss_output = decoder_patches.gather(dim=1, index=mask_indices.unsqueeze(-1).expand(-1, -1, decoder_patches.size(-1)))
       
        # Compute the total loss.
        self.lossfn= self.loss_fn
        total_loss = self.loss_fn(loss_patch, loss_output)

        return total_loss, loss_patch, loss_output

    def forward(self, images):
        total_loss, loss_patch, loss_output = self.calculate_loss(images)
        return total_loss
    

class decoder_out(nn.Module):
    def __init__(self,device, patch_layer, patch_layer2, patch_encoder, encoder, decoder):
        super().__init__()
        
        self.loss_fn = nn.MSELoss().to(device)
        self.patch_layer = patch_layer
        self.patch_layer2 = patch_layer2
        self.patch_encoder = patch_encoder
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def layer_prop(self):
       return self.patch_layer, self.encoder, self.decoder, self.patch_encoder

    def decoder_out(self, images, test=False):
       
        patches = self.patch_layer(images)
        unmasked_embeddings, masked_embeddings, unmasked_positions, mask_indices, unmask_indices = self.patch_encoder(patches)

        encoder_outputs = self.encoder(unmasked_embeddings)

        encoder_outputs = encoder_outputs + unmasked_positions
        decoder_inputs = torch.cat([encoder_outputs, masked_embeddings], dim=1)

        decoder_outputs = self.decoder(decoder_inputs)
        decoder_patches = self.patch_layer2(decoder_outputs)
        

        loss_patch = patches.gather(dim=1, index=mask_indices.unsqueeze(-1).expand(-1, -1, patches.size(-1)))
        loss_output = decoder_patches.gather(dim=1, index=mask_indices.unsqueeze(-1).expand(-1, -1, decoder_patches.size(-1)))
       
        # Compute the total loss.
        self.lossfn= self.loss_fn
        total_loss = self.loss_fn(loss_patch, loss_output)

        return decoder_outputs

    def forward(self, images):
        out = self.decoder_out(images)
        return out