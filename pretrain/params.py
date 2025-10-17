"""
The baseline code is configured with 3 channels to enable effective visualization of the GenENet model.
For final deployment, it can be reduced to 1 channel for model optimization.
"""

import attr

@attr.s(auto_attribs=True)
class GNETParams:
    tensor_directory = "./dataset"    
 
    BUFFER_SIZE: int = 1024
    
    batch_size: int = 256
    input_shape: int= (48, 48, 3)
    num_classes: int = 26

    # OPTIMIZER
    learning_rate : float = 5e-3
    weight_decay : float = 1e-4


    # PRETRAINING
    EPOCHS = 100
    num_epoch : int = 700

    # AUGMENTATION
    image_size : int = 48
    patch_size : int = 6
    num_patches : int = (image_size // patch_size) ** 2

    mask_proportion : int = 0.8125 

    img_size: int = 48
    dropout_rate: float = 0.1

    # ENCODER and DECODER
    layer_norm_eps: float = 1e-6
    enc_proj_dim: int =  128
    dec_proj_dim: int =   64
    ENC_NUM_HEADS = 4
    enc_num_heads : int =  4
    ENC_LAYERS = 6
    enc_num_layers: int = 6
    dec_num_heads : int = 4
    dec_num_layers = (2)

    enc_trans_units = [
    enc_proj_dim * 2,
        enc_proj_dim,
    ]

    dec_trans_units = [
    dec_proj_dim * 2,
        dec_proj_dim,
    ]

