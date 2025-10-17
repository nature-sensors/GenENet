import torchvision.transforms as transforms
from utils.mask_transform import masktransform

    
class DataAugmentation:
    def __init__(self, params):
        self.params = params

    def tensor_to_pil(self, tensor):
        return transforms.ToPILImage()(tensor)
    
    def get_train_augmentation_model(self):
        def transform(x):
            x = self.tensor_to_pil(x)

            x = transforms.Compose([
                transforms.Resize((self.params.input_shape[0] + 20, self.params.input_shape[0] + 20)),
                transforms.RandomCrop(self.params.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])(x)

            return x
        return transform

    def get_test_augmentation_model(self):
        def transform(x):
            x = self.tensor_to_pil(x)

            x = transforms.Compose([
                transforms.Resize((self.params.image_size, self.params.image_size)),
                transforms.ToTensor(),
            ])(x)

            return x
        return transform

    def get_application_test_augmentation_model(self):
        def transform(x):
            x = self.tensor_to_pil(x)

            x = transforms.Compose([
                transforms.Resize((self.params.image_size, self.params.image_size)),
                transforms.ToTensor(),
                masktransform(),
            ])(x)

            return x
        return transform
    
    
