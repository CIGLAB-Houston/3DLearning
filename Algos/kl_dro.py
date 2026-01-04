
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from utils import timer_with_memory
from config import Config
from Net.MISSION_net import DeepLSTM


cfg = Config()
algo_name = 'KLDRO'




class KL_DRO():
    def __init__(self,windows_len,predict_len,epoches,discount_factor,step_size,lr,alpha,budget,device,save_every,display_every,**kwargs):
        self.device = device
        self.LR = lr
        self.STEP_SIZE = step_size
        self.DISCOUNT_FACTOR = discount_factor
        self.ALPHA = torch.tensor(float(alpha), requires_grad=True)
        self.BUDGET = budget
        self.EPOCHS = epoches
        self.WINDOWS_LEN = windows_len
        self.PREDICT_LEN = predict_len
        self.DISPLAY_EVERY = display_every
        self.SAVE_EVERY = save_every
        self.epoches_losses_list = []

    @timer_with_memory
    def train(self, train_data_loader, model_save_folder_path, task=None, model_path=None, **kwargs):
        if model_path:
            model = torch.load(model_path, weights_only=False, map_location=self.device)
        else:
            model = DeepLSTM(output_size=self.PREDICT_LEN).to(self.device)

        print(f'\n=============== {algo_name} Training Start ===============')
        loss_function = nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.LR)
        scheduler = StepLR(optimizer, step_size=self.STEP_SIZE, gamma=self.DISCOUNT_FACTOR)

        for epoch in range(self.EPOCHS):
            epoch_loss = 0
            progress_bar = tqdm(train_data_loader, desc=f"  Iteration [{epoch + 1}/{self.EPOCHS}]", leave=True)

            for batch in progress_bar:
                seq_batch = batch[0].to(self.device)
                time_step = 0


                for t in range(self.WINDOWS_LEN, seq_batch.shape[1] - self.PREDICT_LEN + 1):
                    input_seq = seq_batch[:, t - self.WINDOWS_LEN:t]
                    input_seq = input_seq.unsqueeze(2)
                    target_seq = seq_batch[:, t:t + self.PREDICT_LEN]
                    y_pred = model(input_seq)

                    if task:
                        y_pred = torch.relu(y_pred)
                        y_pred_real = y_pred * (max(kwargs['data_range']) - min(kwargs['data_range'])) + min(
                            kwargs['data_range'])
                        y_real = target_seq * (max(kwargs['data_range']) - min(kwargs['data_range'])) + min(
                            kwargs['data_range'])
                        best_x, lowest_loss = task.find_best_scheduling(y_pred_real)
                        loss = torch.mean(task.loss(x=best_x, y=y_real))

                    else:
                        loss = loss_function(y_pred, target_seq)

                    if torch.isnan(loss) or torch.isinf(loss):
                        print("[Warning] loss is nan or inf, skipping backward")
                        continue

                    batch_loss = torch.exp(loss / self.ALPHA)
                    time_step += 1
                    batch_kl_loss = self.ALPHA * torch.log(batch_loss) + self.ALPHA * self.BUDGET

                    optimizer.zero_grad()
                    batch_kl_loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        self.ALPHA.clamp_(min=1e-6)

                    epoch_loss += batch_kl_loss.item()
                scheduler.step()

            avg_loss = epoch_loss / len(progress_bar)
            progress_bar.set_postfix({'  | Avg Training Loss | ': f'{avg_loss:.4f}'})
            self.epoches_losses_list.append(avg_loss)

            if (epoch + 1) % self.DISPLAY_EVERY == 0:
                print(f'  | Epoch [{epoch + 1}/{self.EPOCHS}] Avg Training Loss | {avg_loss}')

            if (epoch + 1) % self.SAVE_EVERY == 0:
                save_path = cfg.dro_save_model_path(folder=model_save_folder_path, dro_name=algo_name, model_name='ML',
                                                    cuda=self.device, epoch=epoch + 1, lr=self.LR)
                torch.save(model, save_path)
                print(f"  | Model Saved | {save_path}")
        print(f'=============== {algo_name} Training End ===============')


    @timer_with_memory
    def test(self, test_data_loader, model_path):

        model = torch.load(model_path, weights_only=False, map_location=self.device)
        avg_loss = 0
        print(f'\n=============== {algo_name} Testing Start ===============')

        loss_function = nn.MSELoss().to(self.device)

        for epoch in range(1):

            epoch_loss = 0
            time_step = 0

            progress_bar = tqdm(test_data_loader, desc=f"Epoch {epoch + 1}/1", leave=True)
            for i,batch in enumerate(progress_bar):
                seq_batch = batch[0].to(self.device)

                if i == 0:
                    predict_list = []

                for t in range(self.WINDOWS_LEN, seq_batch.shape[1]-self.PREDICT_LEN+1):
                    input_seq = seq_batch[:, t - self.WINDOWS_LEN:t]
                    input_seq = input_seq.unsqueeze(2)
                    target_seq = seq_batch[:, t:t+self.PREDICT_LEN]
                    y_pred = model(input_seq)
                    loss = loss_function(y_pred, target_seq)

                    if i == 0:
                        predict_list.append(y_pred)

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
                seq_batch = batch[0].to(self.device)

                for t in range(self.WINDOWS_LEN, seq_batch.shape[1] - self.PREDICT_LEN + 1):
                    input_seq = seq_batch[:, t - self.WINDOWS_LEN:t]
                    input_seq = input_seq.unsqueeze(2)
                    target_seq = seq_batch[:, t:t + self.PREDICT_LEN]
                    y_pred = torch.relu(model(input_seq))
                    y_pred_real = y_pred * (max(data_range) - min(data_range)) + min(data_range)
                    y_real = target_seq * (max(data_range) - min(data_range)) + min(data_range)

                    x_hat, lowest_loss = task.find_best_scheduling(y=y_pred_real)
                    x_opt, opt_loss = task.find_best_scheduling(y=y_real)
                    evaluate_loss = task.loss(x=torch.tensor(x_hat), y=torch.tensor(y_real))

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


