import os
import sys
import math
import time
import torch
import shutil
import random
import psutil
import datetime
import threading
import tracemalloc
import numpy as np
import pandas as pd
import torch.nn as nn
from functools import wraps
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from scipy.stats import wasserstein_distance
from torch_fidelity import calculate_metrics
from sklearn.preprocessing import MinMaxScaler
from torchvision.transforms.functional import to_pil_image
from datasets import CorruptedDataset




class Utils():
    def __init__(self):
        pass

    def write_dataset(self, data, gen_data_path, title='ContextTokens'):
        df = pd.DataFrame({title: data})
        write_header = True

        if os.path.exists(gen_data_path) and os.path.getsize(gen_data_path) > 0:
            try:
                existing_df = pd.read_csv(gen_data_path, nrows=0)
                existing_columns = existing_df.columns.tolist()

                if title in existing_columns:
                    write_header = False
            except Exception as e:
                print(f"Warning: failed to read existing CSV header. Will write header anyway. Error: {e}")

        df.to_csv(
            gen_data_path,
            mode='a' if not write_header else 'w',
            header=write_header,
            index=False,
            encoding='utf-8'
        )

    def compute_normalized_wasserstein(self,df1, df2):
        x = df1.to_numpy().reshape(-1, 1)
        y = df2.to_numpy().reshape(-1, 1)
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(x).flatten()
        y_scaled = scaler.fit_transform(y).flatten()
        distance = wasserstein_distance(x_scaled, y_scaled)
        return distance

    def draw_loss_plot(self,loss_list,log_folder_path,**kwargs):
        dro_name = None
        diffusion_name = None
        ml_name = None
        for model_name,v in kwargs.items():
            if model_name == 'dro_name':
                dro_name = v
            elif model_name == 'diffusion_name':
                diffusion_name = v
            elif model_name == 'ml_name':
                ml_name = v
            else:
                continue

        partial_title = ''
        for name in [dro_name, diffusion_name, ml_name]:
            if name is None:
                continue
            partial_title += f'-{name}'
        partial_title = partial_title[1:]


        plot_path = os.path.join(log_folder_path, f'{partial_title}_training_loss.png')
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(loss_list) + 1), loss_list, marker='o')
        plt.title(f'{partial_title} Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch+1)]
        entries += [str(meter) for meter in self.meters]

        print('\r' + '\t'.join(entries), end='')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



class TensorDatasetWrapper(Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, idx):
        return self.tensor[idx]

    def __len__(self):
        return self.tensor.shape[0]

def compute_fid_from_tensors(true_images, fake_images, device='cuda:0'):
    def prepare(tensor):
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, dtype=torch.float32)
        #
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)

        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)

        tensor = tensor.clone()
        min_val = tensor.min()
        max_val = tensor.max()
        tensor = (tensor - min_val) / (max_val - min_val + 1e-8)
        tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
        tensor = tensor.to(torch.uint8)
        return tensor

    true_tensor = prepare(true_images)
    fake_tensor = prepare(fake_images)

    dataset1 = TensorDatasetWrapper(true_tensor)
    dataset2 = TensorDatasetWrapper(fake_tensor)

    metrics = calculate_metrics(
        input1=dataset1,
        input2=dataset2,
        fid=True,
        isc=False,
        kid=False,
        input1_model_input_transform=torch.nn.Identity(),
        input2_model_input_transform=torch.nn.Identity(),
        batch_size=64,
        device=torch.device(device),
        input1_cache_name=None,
        input2_cache_name=None,
    )

    return metrics['frechet_inception_distance']

class Evaluation():
    def __init__(self):
        pass

    def fid_evaluation(self,fake_df, true_df,picture_size,device,isshow=True):
        fake_data = fake_df.to_numpy()
        true_data = true_df.to_numpy()

        true_num_images = true_data.shape[0] // (picture_size * picture_size)
        fake_num_images = fake_data.shape[0] // (picture_size * picture_size)

        fake_data = fake_data[:fake_num_images * picture_size * picture_size]
        true_data = true_data[:true_num_images * picture_size * picture_size]
        combined_data = np.concatenate([fake_data, true_data], axis=0)

        combined_data_2d = combined_data.reshape(-1, picture_size)

        combined_data_2d = normalize_rows(X=combined_data_2d)
        combined_data = combined_data_2d.reshape(-1)
        fake_data = combined_data[:len(fake_data)]
        true_data = combined_data[len(fake_data):]

        if isshow:
            plt.plot(true_data[:1000], label='True Data')
            plt.plot(fake_data[:1000],label='Generated Data')
            plt.legend()
            plt.show()

        fake_images = fake_data.reshape(fake_num_images, picture_size, picture_size)
        true_images = true_data.reshape(true_num_images, picture_size, picture_size)
        print(f'  | Fake Images | {fake_images.shape}')
        print(f'  | True Images | {true_images.shape}')

        compute_fid_from_tensors(fake_images=fake_images, true_images=true_images,device=device)



def normalize_rows(X):
    X = np.array(X, dtype=np.float32)
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mean) / std




class Tee:
    def __init__(self, filepath, mode='a'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(filepath, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class Log_Redirect():
    def __init__(self):
        pass

    def build_log_file(self,result_path,cuda):
        result_path = os.path.join(result_path,cuda.replace(':', '_'))

        if os.path.exists(result_path) == False:
            os.mkdir(result_path)

        today = datetime.datetime.now().strftime('%Y-%m-%d')
        rand_num = random.randint(0, 99999)
        log_folder_name = f"{today}_{rand_num}"
        log_filename = f"{today}_{rand_num}.txt"
        log_folder_path = os.path.join(result_path, log_folder_name)
        log_file_path = os.path.join(log_folder_path, log_filename)

        if os.path.exists(log_folder_path):
            shutil.rmtree(log_folder_path)

        os.mkdir(log_folder_path)
        return log_folder_path,log_file_path




def timer_with_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        device = None
        if args and hasattr(args[0], "device"):
            device = args[0].device

        if torch.cuda.is_available():
            if device is not None:
                if isinstance(device, torch.device):
                    torch.cuda.set_device(device)
                elif isinstance(device, int):
                    torch.cuda.set_device(device)
                elif isinstance(device, str) and device.startswith("cuda"):
                    torch.cuda.set_device(torch.device(device))
            else:
                torch.cuda.set_device(torch.cuda.current_device())

        if args and hasattr(args[0], "__class__"):
            cls_name = args[0].__class__.__name__
            func_name = f"{cls_name}.{func.__name__}"
        else:
            func_name = func.__name__

        stop_flag = False
        cpu_samples = []
        gpu_samples = []

        process = psutil.Process(os.getpid())

        def monitor_memory():
            while not stop_flag:

                cpu_rss = process.memory_info().rss / (1024 ** 2)
                cpu_samples.append(cpu_rss)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)
                else:
                    gpu_mem = 0
                gpu_samples.append(gpu_mem)

                time.sleep(0.5)

        monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        monitor_thread.start()

        tracemalloc.start()
        start_time = time.time()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        gpu_start_alloc = torch.cuda.memory_allocated()

        result = func(*args, **kwargs)

        end_time = time.time()
        current, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        torch.cuda.synchronize()
        gpu_end_alloc = torch.cuda.memory_allocated()
        gpu_peak_alloc = torch.cuda.max_memory_allocated()

        stop_flag = True
        monitor_thread.join()

        cpu_avg = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
        cpu_peak = max(cpu_samples) if cpu_samples else 0

        gpu_avg = sum(gpu_samples) / len(gpu_samples) if gpu_samples else 0
        gpu_peak = max(gpu_samples) if gpu_samples else 0

        print(f"  | {func_name} Runtime | {end_time - start_time:.2f} s")
        print(f"  | {func_name} Max CPU Usage | {cpu_peak:.2f} MB")
        print(f"  | {func_name} Avg CPU Usage | {cpu_avg:.2f} MB")
        print(f"  | {func_name} Peak GPU Allocated | {gpu_peak_alloc / 1024**2:.2f} MB")

        return result

    return wrapper

