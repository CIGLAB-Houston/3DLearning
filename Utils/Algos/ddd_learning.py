import math
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
from Utils.datasets import Data_Loader
from Utils.diffusion_utils import DiffusionProcess,diffusion_name
from Utils.Algos.ml import algo_name as ml_name
from Utils.config import Config
from Utils.utils import timer_with_memory


algo_name = '3DLearning'
cfg = Config()



class DDD_Learning():
    def __init__(self, device,discount_factor,batch_size,pic_size,epoches,ppo_clip,display_every,
                 dro_lr,step_size,adjust_timesteps,mu,eta,budget,p_s0,ml_save_every,dfu_save_every,
                 batch_repeat,beta_1,beta_T,T,dfu_name,selected_timesteps,windows_len,predict_len,ml_lr,**kwargs):

        self.device = device
        self.DISCOUNT = discount_factor
        self.BATCH_SIZE = batch_size
        self.PIC_SIZE = pic_size
        self.EPOCHES = epoches
        self.DRO_INNER_EPOCHES = kwargs['dro_inner_epochs']
        self.ML_INNER_EPOCHES = kwargs['ml_inner_epochs']
        self.PPO_CLIP = ppo_clip
        self.DISPLAY_EVERY = display_every
        self.DFU_NAME = dfu_name

        self.DRO_LR = dro_lr
        self.STEP_SIZE = step_size
        self.ADJUST_TIMESTEPS = adjust_timesteps
        self.MU = mu
        self.ETA = eta
        self.BUDGET = budget
        self.P_S0 = p_s0
        self.ML_SAVE_EVERY = ml_save_every
        self.DFU_SAVE_EVERY = dfu_save_every

        self.BATCH_REPEAT = batch_repeat
        self.BETA_1 = beta_1
        self.BETA_T = beta_T
        self.T = T

        self.SELECTED_TIMESTEPS = selected_timesteps[:self.ADJUST_TIMESTEPS]

        self.WINDOWS_LEN = windows_len
        self.PREDICT_LEN = predict_len
        self.ML_LR = ml_lr

        self.betas = torch.linspace(start=self.BETA_1, end=self.BETA_T, steps=self.T).to(device=self.device)
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start=self.BETA_1, end=self.BETA_T, steps=self.T), dim=0).to(device=self.device)
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=self.device), self.alpha_bars[:-1]])
        self.alphas = 1 - self.betas

        self.sigmas = self.sigmas()
        self.epoches_losses_list = []
        self.cur_r_theta = None
        self.next_r_theta = None

    @timer_with_memory
    def train(self ,ml_model_path ,dfu_model_path , z0,s0, z0_trajectory ,model_save_folder_path,task=None,**kwargs):

        def forward_fn(z_t, t_tensor):
            t = int(t_tensor.item())
            return dfu_theta(z_t, t)

        dfu_0 = torch.load(dfu_model_path, weights_only=False, map_location=self.device)
        hw = torch.load(ml_model_path, weights_only=False, map_location=self.device)
        dfu_theta = copy.deepcopy(dfu_0)


        print(f'\n=============== {algo_name} Training Start ===============')
        print(f'  | Selected Timesteps | {self.SELECTED_TIMESTEPS}')

        dfu_opt = torch.optim.Adam(dfu_theta.parameters(), lr=self.DRO_LR)
        dfu_scheduler = StepLR(dfu_opt, step_size=self.STEP_SIZE, gamma=self.DISCOUNT)

        z0_loader = Data_Loader(df=z0).ml_set2loader(
                                                     split_test=False,
                                                     device=self.device,
                                                     batch_size=self.BATCH_SIZE,
                                                     normlized=True,
                                                     pic_size=self.PIC_SIZE
                                                     )
        s0_dl = Data_Loader(df=s0)
        s0_iteration = math.ceil(len(s0_dl.data_group) / (self.PIC_SIZE * self.PIC_SIZE * self.BATCH_SIZE)) * self.BATCH_REPEAT
        s0_dataset = s0_dl.dfu_dataset(
                                        batch_size=self.BATCH_SIZE,
                                        total_iteration=s0_iteration,
                                        normlized=True,
                                        pic_size=self.PIC_SIZE
                                        )

        a0_T = self.build_a0(
                             total_iteration=s0_iteration,
                             z0_trajectory=z0_trajectory,
                             dfu=dfu_0,
                            )

        for epoch_1 in range(self.EPOCHES):

            if task:
                loss_hw = self.loss_hw_task(test_data_loader=z0_loader, model=hw,task=task,**kwargs)
            else:
                loss_hw = self.loss_hw(test_data_loader=z0_loader, model=hw)

            dfu_theta.train()
            for epoch_2 in range(self.DRO_INNER_EPOCHES):
                print(f'  | Epoch [{epoch_1 + 1}/{self.EPOCHES}] - [{epoch_2 + 1}/{self.DRO_INNER_EPOCHES}] | Training DRO')
                s0_dataiterator = s0_dl.dfu_set2inter(dfu_dataset=s0_dataset,batch_size=self.BATCH_SIZE)
                progress_bar = tqdm(range(s0_iteration), desc="    DRO Inner Iterations")
                jsm_loss_total = 0

                for i in progress_bar:
                    s0_data = s0_dataiterator.__next__()
                    s0_data = s0_data[0].to(device=self.device)
                    jsm_loss = dfu_theta.loss_fn(x=s0_data, idx=None, T_prime=self.ADJUST_TIMESTEPS)

                    r_theta = self.r_theta(z0_trajectory=z0_trajectory, a0_T=a0_T,batch_i=i,forward_fn=forward_fn)
                    ppo_loss = self.ppo(r=r_theta, loss_hw=loss_hw, batch_i=i,detach_part='hw')
                    jsm_adjust_loss = self.MU * jsm_loss
                    dro_inner_loss = -(ppo_loss - jsm_adjust_loss)
                    jsm_loss_total += jsm_loss.item()

                    progress_bar.set_postfix(loss=dro_inner_loss.item())
                    dfu_opt.zero_grad()
                    dro_inner_loss.backward()
                    dfu_opt.step()

                jsm_avg_loss = jsm_loss_total / s0_iteration
                self.MU = self.MU + self.ETA * (jsm_avg_loss - self.BUDGET)
                dfu_scheduler.step()


            s_theta = self.gen_s_theta(
                diffusion_model=dfu_theta,
                gen_iterations=1,
            )

            s_theta_df = pd.DataFrame(s_theta)
            s_theta_loader = Data_Loader(df=s_theta_df).ml_set2loader(
                split_test=False,
                device=self.device,
                batch_size=self.BATCH_SIZE,
                normlized=True,
                pic_size=self.PIC_SIZE
            )


            print(f'  | Epoch [{epoch_1 + 1}/{self.EPOCHES} | Training ML based on S_theta')

            hw, epoches_losses = self.train_ml(
                model=hw,
                lr=self.ML_LR,
                epoches=self.ML_INNER_EPOCHES,
                train_data_loader=s_theta_loader,
                task=task,
                **kwargs
            )

            self.epoches_losses_list += epoches_losses

            if (epoch_1+1) % self.DFU_SAVE_EVERY == 0:

                save_path = cfg.dro_save_model_path(
                    folder=model_save_folder_path,
                    dro_name=algo_name,
                    model_name=diffusion_name,
                    cuda=self.device,
                    epoch=(epoch_1+1),
                    lr=self.DRO_LR,
                )
        #
                torch.save(dfu_theta, save_path)
                print(f"  | {algo_name}-{diffusion_name} Model Saved | {save_path}")

        #
            if (epoch_1+1) % self.ML_SAVE_EVERY == 0:

                save_path = cfg.dro_save_model_path(
                    folder=model_save_folder_path,
                    dro_name=algo_name,
                    model_name=ml_name,
                    cuda=self.device,
                    epoch=epoch_1+1,
                    lr=self.ML_LR,
                )
                torch.save(hw, save_path)
                print(f"  | {algo_name}-{ml_name} Model Saved | {save_path}")

        print(f'=============== {algo_name} Training End ===============')

    def sigmas(self):
        sigmas = torch.sqrt((1 - self.alpha_prev_bars) / (1 - self.alpha_bars) * self.betas)
        return sigmas

    def gen_s_theta(self,diffusion_model,gen_iterations):
        diffusion_model.eval()
        s_theta = []

        for _ in range(gen_iterations):
            process = DiffusionProcess(
                beta_1=self.BETA_1,
                beta_T=self.BETA_T,
                T=self.T,
                diffusion_fn=diffusion_model,
                device=self.device,
                shape=(1, self.PIC_SIZE, self.PIC_SIZE),
            )

            sample, _ = process.sampling(
                                         sampling_number=self.BATCH_SIZE,
                                         save_trajectory=False
                                        )

            generate_data = sample.detach().cpu().numpy().reshape(-1)
            s_theta.append(generate_data)

        s_theta = np.concatenate(s_theta, axis=0)

        return s_theta


    def build_a0(self,total_iteration,z0_trajectory,dfu):
        a0_T = []
        for i in range(total_iteration):
            a0_T_i = []

            for idx,t in enumerate(self.SELECTED_TIMESTEPS): # 4

                z_t = z0_trajectory[i][idx]
                z_t_1 = z0_trajectory[i][idx - 1]

                if t == 0:
                    z_t_1 = z0_trajectory[i][idx]

                epsilon = dfu(z_t, t)

                mu_dfu0_t = self.mu_t(x_t=z_t, t=t,epsilon=epsilon).detach()

                a0_t = (z_t_1 - mu_dfu0_t) ** 2


                a0_T_i.append(a0_t)

            a0_T_i = torch.stack(a0_T_i, dim=0)
            a0_T.append(a0_T_i)
        a0_T = torch.stack(a0_T, dim=0)
        return a0_T


    def mu_t(self,x_t,t,epsilon):
        mu = torch.sqrt(1 / self.alphas[t]) * (x_t - self.betas[t] / torch.sqrt(1 - self.alpha_bars[t]) * epsilon)
        return mu


    def loss_hw(self, test_data_loader, model):
        avg_mse_list = []

        for batch in test_data_loader:
            seq_batch = batch[0].to(self.device)
            target_mat = []
            y_pred_mat = []

            for t in range(self.WINDOWS_LEN, seq_batch.shape[1] - self.PREDICT_LEN+1):
                input_seq = seq_batch[:, t - self.WINDOWS_LEN:t]
                input_seq = input_seq.unsqueeze(2)
                target_seq = seq_batch[:, t:t+self.PREDICT_LEN]
                y_pred = model(input_seq)
                target_mat.append(target_seq)

                y_pred_mat.append(y_pred)

            target_mat_tensor = torch.cat(target_mat, dim=1)
            y_pred_mat_tensor = torch.cat(y_pred_mat, dim=1)
            batch_mse = torch.mean((y_pred_mat_tensor - target_mat_tensor) ** 2, dim=1, keepdim=True)

            avg_mse_list.append(batch_mse)

        avg_mse_tensor = torch.cat(avg_mse_list, dim=1)

        return avg_mse_tensor

    def loss_hw_task(self, test_data_loader, model,task,**kwargs):

        avg_regret_list = []

        for batch in test_data_loader:
            seq_batch = batch[0].to(self.device)
            regret_mat = []

            for t in range(self.WINDOWS_LEN, seq_batch.shape[1] - self.PREDICT_LEN+1):
                input_seq = seq_batch[:, t - self.WINDOWS_LEN:t]
                input_seq = input_seq.unsqueeze(2)
                target_seq = seq_batch[:, t:t+self.PREDICT_LEN]

                y_pred = torch.relu(model(input_seq))
                y_pred_real = y_pred * (max(kwargs['data_range']) - min(kwargs['data_range'])) + min(kwargs['data_range'])
                y_real = target_seq * (max(kwargs['data_range']) - min(kwargs['data_range'])) + min(kwargs['data_range'])

                x_hat, lowest_loss = task.find_best_scheduling(y=y_pred_real)
                x_opt, opt_loss = task.find_best_scheduling(y=y_real)
                evaluate_loss = task.loss(x=x_hat, y=y_real)
                regret = evaluate_loss - opt_loss
                regret_mat.append(regret)

            regret_mat_tensor = torch.cat(regret_mat, dim=1)
            batch_regret = torch.mean(regret_mat_tensor, dim=1, keepdim=True)
            avg_regret_list.append(batch_regret)

        avg_regret_tensor = torch.cat(avg_regret_list, dim=1)
        return avg_regret_tensor

    def r_theta(self,z0_trajectory,a0_T,batch_i,forward_fn):
        a_diff = torch.zeros(self.BATCH_SIZE, self.PIC_SIZE).to(self.device)
        sum_a = torch.zeros(self.BATCH_SIZE, self.PIC_SIZE).to(self.device)

        for idx,t in enumerate(self.SELECTED_TIMESTEPS):

            if idx == 0:
                continue

            z_t = z0_trajectory[batch_i][idx]
            z_t_1 = z0_trajectory[batch_i][idx - 1]

            t_tensor = torch.tensor(t, device=self.device)
            epsilon = checkpoint(forward_fn, z_t, t_tensor, use_reentrant=False)

            mu_dfu_t = self.mu_t(x_t=z_t, t=t, epsilon=epsilon)

            a = (z_t_1 - mu_dfu_t) ** 2
            a = a.sum(dim=-1)
            a = a.squeeze(1)

            a0 = a0_T[batch_i][idx].to(device=self.device)
            a0 = a0.sum(dim=-1)
            a0 = a0.squeeze(1)

            a_diff += (a - a0.detach()) / (2 * (self.sigmas[t] ** 2))
            sum_a += a
        r = torch.exp(-a_diff)


        return r

    def ppo(self,r,loss_hw,batch_i,detach_part):

        if detach_part == 'hw':
            loss_hw = loss_hw.detach()

        elif detach_part == 'r':
            r = r.detach()

        unclipped = r * loss_hw[:, batch_i * self.PIC_SIZE:(batch_i + 1) * self.PIC_SIZE]
        clipped_r = torch.clamp(r, 1 - self.PPO_CLIP, 1 + self.PPO_CLIP)
        clipped = clipped_r * loss_hw[:, batch_i * self.PIC_SIZE:(batch_i + 1) * self.PIC_SIZE]
        ppo_loss = torch.min(unclipped, clipped).mean()
        return ppo_loss

    @timer_with_memory
    def test(self, test_data_loader, model_path):

        model = torch.load(model_path, weights_only=False, map_location=self.device)
        avg_loss = 0
        print(f'\n=========== {algo_name} Testing Start ===========')

        loss_function = nn.MSELoss().to(self.device)

        for epoch in range(1):

            epoch_loss = 0
            time_step = 0
            progress_bar = tqdm(test_data_loader, desc=f"Epoch {epoch + 1}/1", leave=True)
            for i,batch in enumerate(progress_bar):
                seq_batch = batch[0].to(self.device)


                for t in range(self.WINDOWS_LEN, seq_batch.shape[1]-self.PREDICT_LEN+1):
                    input_seq = seq_batch[:, t - self.WINDOWS_LEN:t]
                    input_seq = input_seq.unsqueeze(2)
                    target_seq = seq_batch[:, t:t+self.PREDICT_LEN]
                    y_pred = model(input_seq)
                    loss = loss_function(y_pred, target_seq)

                    epoch_loss += loss.item()
                    time_step += 1

            avg_loss = epoch_loss / time_step
            progress_bar.set_postfix({'  | Avg Testing Loss | ': f'{avg_loss:.4f}'})

            print(f'  | Epoch [{epoch + 1}/1] Avg Testing Loss | {avg_loss}')

        print(f'=============== {algo_name} Testing End ===============')
        return avg_loss

    @timer_with_memory
    def test_task(self, test_data_loader, model_path, task, data_range):

        model = torch.load(model_path, weights_only=False, map_location=self.device)

        print(f'\n=============== {algo_name} Testing Start ===============')

        loss_function = nn.MSELoss().to(self.device)

        for epoch in range(1):

            epoch_loss = 0
            epoch_loss2 = 0
            epoch_loss3 = 0
            time_step = 0
            progress_bar = tqdm(test_data_loader, desc=f"Epoch {epoch + 1}/1", leave=True)
            for i, batch in enumerate(progress_bar):
                seq_batch = batch[0].to(self.device)  # [batch_size, sequence_length]）

                for t in range(self.WINDOWS_LEN, seq_batch.shape[1] - self.PREDICT_LEN + 1):
                    input_seq = seq_batch[:, t - self.WINDOWS_LEN:t]  # [batch_size, window_len]
                    input_seq = input_seq.unsqueeze(2)
                    target_seq = seq_batch[:, t:t + self.PREDICT_LEN]  # [batch_size, 1]
                    y_pred = torch.relu(model(input_seq))
                    y_pred_real = y_pred * (max(data_range) - min(data_range)) + min(data_range)
                    y_real = target_seq * (max(data_range) - min(data_range)) + min(data_range)

                    x_hat, lowest_loss = task.find_best_scheduling(y=y_pred_real)
                    x_opt, opt_loss = task.find_best_scheduling(y=y_real)
                    evaluate_loss = task.loss(
                        x=x_hat,
                        y=y_real
                    )
                    regret = torch.mean(evaluate_loss - opt_loss)
                    norm_regret = regret / (-torch.mean(opt_loss))
                    loss = loss_function(y_pred, target_seq)
                    epoch_loss += loss.item()
                    epoch_loss2 += regret.item()
                    epoch_loss3 += norm_regret.item()
                    time_step += 1

            avg_loss = epoch_loss / time_step
            avg_task_loss = epoch_loss2 / time_step
            avg_norm_task_loss = epoch_loss3 / time_step
            progress_bar.set_postfix({'  | Avg Testing Loss | ': f'{avg_loss:.4f}'})

            print(
                f'  | Epoch [{epoch + 1}/1] Avg MSE Loss | {avg_loss} Avg Task Regret | {avg_task_loss} Avg Task Norm Regret | {avg_norm_task_loss}')
        print(f'=============== {algo_name} Testing End ===============')



    def train_ml(self, train_data_loader, model,lr,epoches,task=None,**kwargs):
        loss_function = nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=self.STEP_SIZE, gamma=self.DISCOUNT)
        epoches_losses = []

        for epoch in range(epoches):
            epoch_loss = 0
            time_step = 0
            progress_bar = tqdm(train_data_loader, desc=f"Epoch {epoch + 1}/{epoches}", leave=True)

            for batch in progress_bar:

                seq_batch = batch[0].to(self.device)

                for t in range(self.WINDOWS_LEN, seq_batch.shape[1]-self.PREDICT_LEN+1):
                    input_seq = seq_batch[:, t - self.WINDOWS_LEN:t]
                    input_seq = input_seq.unsqueeze(2)
                    target_seq = seq_batch[:, t:t + self.PREDICT_LEN]

                    y_pred = model(input_seq)

                    if task:
                        y_pred = torch.relu(y_pred)
                        y_pred_real = y_pred * (max(kwargs['data_range']) - min(kwargs['data_range'])) + min(kwargs['data_range'])
                        y_real = target_seq * (max(kwargs['data_range']) - min(kwargs['data_range'])) + min(kwargs['data_range'])
                        best_x, lowest_loss = task.find_best_scheduling(y_pred_real)
                        loss = torch.mean(task.loss(x=best_x, y=y_real))

                    else:
                        loss = loss_function(y_pred, target_seq)

                    if torch.isnan(loss) or torch.isinf(loss):
                        print("[Warning] loss is nan or inf, skipping backward")
                        continue

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    time_step += 1
            scheduler.step()

            avg_loss = epoch_loss / time_step
            progress_bar.set_postfix({'  | Avg Training Loss | ': f'{avg_loss:.4f}'})
            epoches_losses.append(avg_loss)

            print(f'  | Epoch [{epoch + 1}/{epoches}] Avg Training Loss | {avg_loss}')

        return model,epoches_losses

#














