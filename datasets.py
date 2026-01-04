import numpy as np
import torch
from torch.utils.data import Dataset,TensorDataset,DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

from PIL import Image
from perlin_noise import PerlinNoise







class Data_Loader():
    def __init__(self,df,len_limit=None):
        self.data_group = df.to_numpy()
        if len_limit:
            self.data_group = self.data_group[:min(len_limit,len(self.data_group))]

        self.data_len = 0

    def data_transfer(self, normlized=True, pic_size=28, noise_type=None, noise_param=0.05):

        flattened_data = np.concatenate([arr.flatten() for arr in self.data_group])

        has_nan = np.isnan(flattened_data).any()
        if has_nan:
            mean_value = np.nanmean(flattened_data)
            flattened_data = np.nan_to_num(flattened_data, nan=mean_value)

        if normlized:
            max_value = np.max(flattened_data)
            min_value = np.min(flattened_data)
            if max_value == min_value:
                normalized_data = np.zeros_like(flattened_data)
            else:
                normalized_data = (flattened_data - min_value) / (max_value - min_value)
        else:
            normalized_data = flattened_data


        if noise_type:

            if noise_type == 'gaussian':
                print(f'  | Noise | {noise_type} noise - {noise_param}')
                noise = np.random.normal(loc=0.0, scale=noise_param, size=normalized_data.shape)
                normalized_data = normalized_data + noise
                normalized_data = np.clip(normalized_data, 0.0, 1.0)

            elif noise_type == 'perlin':
                print(f'  | Noise | {noise_type} noise - {noise_param}')

                noise = PerlinNoise(octaves=8)
                perlin = np.array([noise(i / len(normalized_data)) for i in range(len(normalized_data))])
                perlin = perlin - perlin.mean()
                perlin = perlin / (np.abs(perlin).max() + 1e-8)
                normalized_data = normalized_data + noise_param * perlin
                normalized_data = np.clip(normalized_data, 0.0, 1.0)

            elif noise_type == 'cutout':
                print(f'  | Noise | {noise_type} noise - {noise_param}')
                length = int(noise_param * len(normalized_data))
                start = np.random.randint(0, len(normalized_data) - length)
                normalized_data[start:start + length] = 1

        remainder = len(normalized_data) % pic_size
        if remainder != 0:
            normalized_data = normalized_data[:-remainder]

        dataset = normalized_data.reshape(-1, pic_size)
        self.data_len = len(dataset)
        return dataset

    def ml_dataset(self,normlized=True,pic_size=28,noise_type=None, noise_param=0.05):
        dataset = self.data_transfer(normlized=normlized,pic_size=pic_size, noise_type=noise_type, noise_param=noise_param)
        train_data, test_data = train_test_split(dataset, test_size=0.3, random_state=42, shuffle=False)
        return train_data, test_data

    def dfu_dataset(self,batch_size, total_iteration,normlized=True,pic_size=28):
        data_transfer = self.data_transfer(normlized=normlized,pic_size=pic_size)
        num_days, num_features = data_transfer.shape
        split_size = pic_size

        num_full_groups = num_days // split_size
        remaining_days = num_days % split_size

        grouped_data = np.array(
            np.split(data_transfer[:num_full_groups * split_size], num_full_groups))
        if remaining_days > 0:
            missing_days = split_size - remaining_days
            last_group = np.vstack(
                [data_transfer[num_full_groups * split_size:], data_transfer[:missing_days]])
            grouped_data = np.vstack([grouped_data, last_group[np.newaxis, :, :]])

        grouped_data = np.expand_dims(grouped_data, axis=1)

        total_images = batch_size * total_iteration

        num_original_images = grouped_data.shape[0]

        repeat_times = (total_images // num_original_images) + 1

        expanded_data = np.tile(grouped_data, (repeat_times, 1, 1, 1))
        expanded_data = expanded_data[:total_images]

        expanded_data_tensor = torch.tensor(expanded_data, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(expanded_data_tensor)

        return dataset

    def ml_set2loader(self,split_test,device,batch_size,normlized=True,pic_size=28,noise_type=None, noise_param=0.05):
        if split_test:
            train_data, test_data = self.ml_dataset(pic_size=pic_size,normlized=normlized,noise_type=noise_type, noise_param=noise_param)
            print('\n  | Train Dataset | ', train_data.shape)
            print('  | Test  Dataset | ', test_data.shape)
            train_data = torch.FloatTensor(train_data).to(device)
            test_data = torch.FloatTensor(test_data).to(device)
            train_dataset = TensorDataset(train_data)
            test_dataset = TensorDataset(test_data)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
            return train_loader, test_loader
        else:
            data = self.data_transfer(pic_size=pic_size,normlized=normlized,noise_type=noise_type, noise_param=noise_param)
            print('\n  | Whole Dataset | ', data.shape)
            data = torch.FloatTensor(data).to(device)
            dataset = TensorDataset(data)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
            return data_loader

    def dfu_set2inter(self,dfu_dataset,batch_size):
        dataloader = DataLoader(dfu_dataset, batch_size=batch_size, drop_last=True)
        dataiterator = iter(dataloader)
        return dataiterator

class CorruptedDataset(Dataset):
    def __init__(self, base_dataset, corruption_fn,corrupt_label, normalization,severity=5):
        self.base_dataset = base_dataset
        self.corruption_fn = corruption_fn
        self.severity = severity
        self.corrupt_label = corrupt_label
        self.normalization = normalization

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, label = self.base_dataset[index]
        if isinstance(image, torch.Tensor):
            image = T.ToPILImage()(image)

        corrupted_np = self.corruption_fn(image)

        corrupted_img = T.ToTensor()(Image.fromarray(corrupted_np.astype(np.uint8)))
        if self.normalization:

            corrupted_img = (corrupted_img + 0.1307) * 0.3081

        if self.corrupt_label:
            noisy_label = label + np.random.normal(loc=0.0, scale=1.0)
            label = int(np.round(noisy_label))
            label = np.clip(label, 0, 9)

        return corrupted_img, label









