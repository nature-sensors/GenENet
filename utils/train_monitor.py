
import torch

def trainmonitor(model, device, epoch, testimage):
    testimage=testimage.to(device)
    (patch_layer, encoder, decoder, patch_encoder)=model.layer_prop()
    
    test_patches = patch_layer(testimage)

    (
                test_unmasked_embeddings,
                test_masked_embeddings,
                test_unmasked_positions,
                test_mask_indices,
                test_unmask_indices,
            ) = patch_encoder(test_patches)
    
    test_encoder_outputs = encoder(test_unmasked_embeddings)
    test_encoder_outputs = test_encoder_outputs + test_unmasked_positions
    test_decoder_inputs = torch.cat(
        [test_encoder_outputs, test_masked_embeddings], dim=1
    )
    test_decoder_outputs = decoder(test_decoder_inputs)

    test_masked_patch, idx = patch_encoder.generate_masked_image(
        test_patches, test_unmask_indices
    )
    print("Idx chosen: ", idx)
    original_image = testimage[idx]
    masked_image = patch_layer.reconstruct_from_patch(
        test_masked_patch
    )
    reconstructed_image = test_decoder_outputs[idx]

    return original_image.cpu().permute(1,2,0), masked_image.cpu(), reconstructed_image.cpu()


