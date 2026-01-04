
class Config(object):
    def __init__(self):
        pass

    def diffusion_model_path(self,folder,model_name,cuda,epoch,batch_size,pic_size,T,lr,beta_1,beta_T,**kwargs):
        model_path = f'{folder}/model_{model_name}_cuda-{cuda.split(":")[1]}_epoch-{epoch}_batch-{batch_size}_picsize-{pic_size}_T-{T}_lr-{lr}_beta1-{beta_1}_betaT-{beta_T}.pth'
        return model_path


    def generate_data_path(self,folder,model_name,cuda,epoch,batch_size,pic_size,T,beta_1,beta_T,**kwargs):

        data_path = rf'{folder}/data_{model_name}_cuda-{cuda.split(":")[1]}_epoch-{epoch}_batch-{batch_size}_picsize-{pic_size}_T-{T}_beta1-{beta_1}_betaT-{beta_T}.csv'

        return data_path


    def ml_model_path(self,folder,model_name,cuda,epoch,lr,**kwargs):
        model_path = rf'{folder}/model_{model_name}_cuda-{cuda.split(":")[1]}_epoch-{epoch}_lr-{lr}.pth'
        return model_path

    def trajectory_data_path(self,folder,model_name,cuda,epoch,batch_size,pic_size,T,beta_1,beta_T,**kwargs):
        data_path = rf'{folder}/data_{model_name}_cuda-{cuda.split(":")[1]}_epoch-{epoch}_batch-{batch_size}_picsize-{pic_size}_T-{T}_beta1-{beta_1}_betaT-{beta_T}_trajectory.npy'

        return data_path



    def dro_save_model_path(self,folder,dro_name,model_name,cuda,epoch,lr,**kwargs):
        model_path = rf'{folder}/model_{dro_name}-{model_name}_cuda-{cuda.split(":")[1]}_epoch-{epoch}_lr-{lr}.pth'
        return model_path




