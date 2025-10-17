"""
The baseline code is configured with 3 channels to enable effective visualization of the GenENet model.
For final deployment, it can be reduced to 1 channel for model optimization.
"""

import torch
import torch.nn as nn
import numpy as np
from models.modules import Patches


class PatchEncoder(nn.Module):
    def __init__(self,params,device,downstream=False):
        super().__init__()
        self.downstream = downstream
        self.params = params
        self.device = device
        (_, self.num_pat, self.patch_area)=Patches(params)(torch.ones((1,3,params.image_size,params.image_size))).shape
        self.projection = nn.Linear(self.patch_area, params.enc_proj_dim)
        self.position_embedding = nn.Embedding(self.num_pat, params.enc_proj_dim)
        self.num_mask = int(params.mask_proportion * self.num_pat)

        self.mask_token = nn.Parameter(torch.randn(1, params.patch_size * params.patch_size * 3))

    def forward(self, patches):

        batch_size = patches.shape[0]
        positions = torch.arange(self.num_pat, device=self.position_embedding.weight.device)

        pos_embeddings = self.position_embedding(positions[None, ...])
        pos_embeddings = pos_embeddings.repeat(batch_size, 1, 1) 
        pos_embeddings= pos_embeddings.to(self.device)
        

        self.projection=self.projection.to(self.device)
        patch_embeddings = self.projection(patches) + pos_embeddings
        

        if self.downstream:
            return patch_embeddings
        else:
            mask_indices, unmask_indices = self.get_random_indices(batch_size)
            mask_indices=mask_indices.to(self.device)
            unmask_indices=unmask_indices.to(self.device)
           
            unmasked_embeddings = patch_embeddings.gather(dim=1, index=unmask_indices.unsqueeze(-1).expand(-1, -1, patch_embeddings.size(-1)))
         
            unmasked_positions = pos_embeddings.gather(dim=1, index=unmask_indices.unsqueeze(-1).expand(-1, -1, patch_embeddings.size(-1)))
            masked_positions = pos_embeddings.gather(dim=1, index=mask_indices.unsqueeze(-1).expand(-1, -1, patch_embeddings.size(-1)))
            
            mask_tokens = self.mask_token.repeat(self.num_mask, 1).unsqueeze(0).repeat(batch_size, 1, 1)
            mask_tokens= mask_tokens.to(self.device)
            
            masked_embeddings = self.projection(mask_tokens) + masked_positions


            return (
                unmasked_embeddings,  
                masked_embeddings,  
                unmasked_positions,  
                mask_indices,  
                unmask_indices,  
            )


    def get_random_indices(self, batch_size):

        rand_indices = torch.argsort(
            torch.rand((batch_size, self.num_pat)), dim=-1
        )
        mask_indices = rand_indices[:, : self.num_mask]
        unmask_indices = rand_indices[:, self.num_mask :]
        return mask_indices, unmask_indices

    def generate_masked_image(self, patches, unmask_indices):
        idx = np.random.choice(patches.shape[0])
        patch = patches[idx]
        unmask_index = unmask_indices[idx]

        new_patch = torch.zeros_like(patch)

        count = 0
        for i in range(unmask_index.shape[0]):
            new_patch[unmask_index[i]] = patch[unmask_index[i]]
        return new_patch, idx