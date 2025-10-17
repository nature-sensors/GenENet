import torch
        
class masktransform(object):

    def __init__(self, maskprob=0.1875):
        self.maskprob = maskprob

    def __call__(self, sample):
        img=sample

        num_rows = int(img.shape[1] * self.maskprob)
        mask = torch.cat((torch.ones(num_rows, img.shape[2]), torch.zeros(img.shape[1] - num_rows, img.shape[2])), dim=0)
        mask = mask.unsqueeze(0).repeat(img.shape[0], 1, 1)
        img = img * mask
        return img
    
