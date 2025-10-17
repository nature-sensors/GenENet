import torch
from torch.utils.data import Dataset
import os


class CustomDataset(Dataset):
    def __init__(self, tensors, transform=None, train=True):
        self.tensors = tensors
        self.transform = transform
        self.train = train

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]

        if self.transform:
            if self.train:
                x = self.transform['train'](x)
            else:
                x = self.transform['test'](x)

        return x, y

    def __len__(self):
        return len(self.tensors[0])
    

class GetTensorDataset:
    def __init__(self, params):
        self.data_tensor = None
        self.label_array = None
        self.params = params

    def load_tensors(self):
        filenames = os.listdir(self.params.tensor_directory)
        data = []
        label_list= list(range(self.params.num_classes))

        for i, filename in enumerate(filenames):
            tensor = torch.load(os.path.join(self.params.tensor_directory, filename))
            data.append(tensor)

        min_shape = [min(t.shape[i] for t in data) for i in range(len(data[0].shape))]
        cropped_batch = [torch.tensor(t[:min_shape[0], :, :]) for t in data]

        self.data_tensor = torch.stack(cropped_batch)
        input_num=self.data_tensor.shape[1]
        self.data_tensor = self.data_tensor.view(-1,32,32,3)
        output_list = [num for item in label_list for num in [item]*input_num]
        self.label_array = torch.tensor(output_list)
        self.data_tensor = self.data_tensor.permute(0, 3, 1, 2).to(torch.float32)


    def get_dataset(self):
        if self.data_tensor is None or self.label_array is None:
            self.load_tensors()
        return self.data_tensor, self.label_array


class GetTensorDataset_Threshold_Sequence(Dataset):
    def __init__(self, params, sequence_length=5, transform=None, train=True):
        self.params = params
        self.sequence_length = sequence_length
        self.transform = transform
        self.train = train
        self.data = []
        self.labels = []

        self.load_tensors() 

    def load_tensors(self):
        filenames = os.listdir(self.params.tensor_directory)
        all_tensors = []
        label_list = list(range(self.params.num_classes))

        for filename in filenames:
            label = int(filename.split('.')[0])  
            tensors = torch.load(os.path.join(self.params.tensor_directory, filename))
            
            for tensor in tensors:
                tensor = torch.tensor(tensor).permute(2, 0, 1) 
                average_value = tensor.view(tensor.shape[0], -1).abs().mean()
                if average_value > self.params.threshold:
                    all_tensors.append((tensor, label))

        for i in range(len(all_tensors) - self.sequence_length):
            sequence = [all_tensors[i + j][0] for j in range(self.sequence_length)]
            transformed_sequence = [self.apply_transform(img) for img in sequence]
            sequence_tensor = torch.stack(transformed_sequence)
            label = all_tensors[i + self.sequence_length - 1][1]
            self.data.append(sequence_tensor)
            self.labels.append(label_list[label])


    def apply_transform(self, img_tensor):
        if self.transform:
            transform_type = 'train' if self.train else 'test'
            img_tensor = self.transform[transform_type](img_tensor)
        return img_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
