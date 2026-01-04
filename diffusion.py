import os
import math
import numpy as np
import torch

from utils import AverageMeter,ProgressMeter,Utils
from datasets import Data_Loader
from Diffusion.DDPM import Model,DiffusionProcess

diffusion_name = 'DDPM'



class Train_DFU():
    def __init__(self,T,beta_1, beta_T, batch_size, lr, total_iteration, save_every, picture_size,
                 device, model_save_folder_path, data_df, **kwargs):
        self.T = T
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.batch_size = batch_size
        self.lr = lr
        self.total_iteration = total_iteration
        self.save_every = save_every
        self.picture_size = picture_size
        self.device = device
        self.model_save_folder_path = model_save_folder_path
        self.data_df = data_df

        self.only_final = True
        self.current_iteration = 0

        self.losses = AverageMeter('Loss', ':.4f')
        self.progress = ProgressMeter(self.total_iteration, [self.losses], prefix='  Iteration ')

        self.env_dataset = Data_Loader(df=self.data_df)

        if not os.path.exists(self.model_save_folder_path):
            os.makedirs(self.model_save_folder_path)


    def train_dataset(self):
        train_dataset = self.env_dataset.dfu_dataset(batch_size=self.batch_size,total_iteration=self.total_iteration,pic_size=self.picture_size)
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, drop_last=True)
        dataiterator = iter(dataloader)
        return dataiterator,dataloader

    def train(self):
        print('\n=============== Diffusion Training Start ===============')
        dataiterator, dataloader = self.train_dataset()
        self.model = Model(device=self.device,
                           beta_1 = self.beta_1,
                           beta_T = self.beta_T,
                           T = self.T)#
        print(f"  | Used Single GPU | {self.device}")

        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        while self.current_iteration != self.total_iteration:
            self.model.train()

            try:
                data = dataiterator.__next__()
            except:
                dataiterator = iter(dataloader)
                data = dataiterator.__next__()

            data = data[0].to(device=self.device)

            loss = self.model.loss_fn(x=data)

            optim.zero_grad()
            loss.backward()
            optim.step()

            self.losses.update(loss.item())
            self.progress.display(self.current_iteration)

            self.current_iteration += 1

            if self.current_iteration % self.save_every == 0:
                save_path = rf'{self.model_save_folder_path}/model_{diffusion_name}_cuda-{self.device.split(":")[1]}_epoch-{self.current_iteration}_batch-{self.batch_size}_picsize-{self.picture_size}_T-{self.T}_lr-{self.lr}_beta1-{self.beta_1}_betaT-{self.beta_T}.pth'

                torch.save(self.model, save_path)
                print(f"  | Model Saved | {save_path}")

        print('=============== Diffusion Training End ===============')


class Inference_DFU():
    def __init__(self,T, beta_1, beta_T, picture_size,batch_size,sampling_number,device,**kwargs):
        self.T = T
        self.BETA_1 = beta_1
        self.BETA_T = beta_T
        self.PICTURE_SIZE = picture_size

        self.BATCH_SIZE = batch_size
        self.SAMPLING_NUMBER = sampling_number
        self.DEVICE = device

        self.utils = Utils()
        self.device = torch.device(self.DEVICE) if torch.cuda.is_available() else torch.device('cpu')



    def generate_wholeset_data(self, save_gen_data_path=None, data_df=None, model_path=None,save_trajectory=False,batch_repeat=1,title='ContextTokens'):
        if save_gen_data_path:
            if os.path.exists(save_gen_data_path):
                os.remove(save_gen_data_path)

        self.env_dataset = Data_Loader(df=data_df)
        n = len(self.env_dataset.data_group)

        total_iteration = math.ceil(n / (self.PICTURE_SIZE * self.PICTURE_SIZE * self.BATCH_SIZE)) * batch_repeat

        print(f'\n=============== Dataset Generating Start ===============')
        print(f'  | Model File | {model_path}')
        print(f"  | Used GPU | {self.device}")
        print(f"  | Number | {n}")
        print(f"  | Totaol Iteration | {total_iteration}")
        print(f"\n  ***** Sampling *****")

        if model_path:
            self.model = torch.load(model_path, map_location=self.device, weights_only=False)

        else:
            self.model = Model(device=self.device,
                               beta_1=self.BETA_1,
                               beta_T=self.BETA_T,
                               T=self.T)
        self.model.eval()

        all_sample_trajectory_data = []
        for i in range(total_iteration):

            process = DiffusionProcess(
                                       beta_1=self.BETA_1,
                                       beta_T=self.BETA_T,
                                       T=self.T,
                                       diffusion_fn=self.model,
                                       device=self.device,
                                       shape=(1, self.PICTURE_SIZE, self.PICTURE_SIZE),
                                       )


            sample,sample_trajectory = process.sampling(sampling_number=self.SAMPLING_NUMBER, save_trajectory=save_trajectory)

            generate_data = sample.detach().cpu().numpy().reshape(-1)
            if save_gen_data_path:
                self.utils.write_dataset(data=generate_data, gen_data_path=save_gen_data_path,title=title)
                if i == total_iteration - 1:
                    print(f"  | Generate Data Saved | {save_gen_data_path}")

            if save_trajectory:
                sample_trajectory = sample_trajectory.detach().cpu().numpy()
                sample_trajectory = np.flip(sample_trajectory, axis=0)
                all_sample_trajectory_data.append(sample_trajectory)

        if save_trajectory:
            all_sample_trajectory_data = np.stack(all_sample_trajectory_data, axis=0)
            print('  | Trajectory Shape | ', all_sample_trajectory_data.shape)
            save_trajectory_filename = os.path.splitext(os.path.basename(save_gen_data_path))[0]
            parent_dir = os.path.dirname(save_gen_data_path)
            save_trajectory_path = os.path.join(parent_dir, f'{save_trajectory_filename}_trajectory.npy')
            np.save(file=save_trajectory_path, arr=all_sample_trajectory_data)
            print(f"  | Generate Trajectory Saved | {save_trajectory_path}")

        print("=============== Dataset Generating End ===============")



