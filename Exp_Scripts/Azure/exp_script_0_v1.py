
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import pandas as pd
import numpy as np
import torch

from diffusion import Train_DFU,Inference_DFU,diffusion_name
from Algos.dag_dro_v1 import DAG_DRO_V1 as DAG_DRO
from Algos.dag_dro_v1 import algo_name as dag_name
from Algos.fw_dro import FW_DRO
from Algos.fw_dro import algo_name as fw_name
from Algos.kl_dro import KL_DRO
from Algos.kl_dro import algo_name as kl_name
from Algos.ml import ML
from Algos.ml import algo_name as ml_name
from Dowstream.dc_load_scheduling import DC_Scheduling
from utils import Utils,Evaluation,Log_Redirect,Tee
from datasets import Data_Loader
from config import Config
torch.cuda.empty_cache()
cfg=Config()



project_root = '/home/jwen5/Publication Code/3D-Learning'


shared_params = {
            'DEVICE': 'cuda:0',
            'PIC_SIZE': 28,
            'MODEL_SAVE': rf'{project_root}/Models/Azure',
            'DATA_FOLDER': rf'{project_root}/Data/Azure',
            'GEN_DATA_SAVE': rf'{project_root}/Gen_Data/Azure',
            'RESULT_SAVE': rf'{project_root}/Results',
            'TRAIN_FILE': rf"{project_root}/Data/Azure/AzureLLMInferenceTrace_2023_Conversation_compressed-5s.csv",
            'TEST_SET': {
                        '24_Conv_5s': rf'{project_root}/Data/Azure/AzureLLMInferenceTrace_2024_Conversation_compressed-5s.csv',

                        '23_Conv_5s': rf'{project_root}/Data/Azure/AzureLLMInferenceTrace_2023_Conversation_compressed-5s.csv',

                        '24_Code_5s': rf'{project_root}/Data/Azure/AzureLLMInferenceTrace_2024_Coding_compressed-5s.csv',

                         '23_Code_5s':rf'{project_root}/Data/Azure/AzureLLMInferenceTrace_2023_Coding_compressed-5s.csv',

                        '23_Code_24_Code_5s': rf'{project_root}/Data/Azure/Combined_AzureLLMInferenceTrace_2023_Coding_2024_Coding_compressed-5s.csv',

                        '23_Code_24_Conv_5s': rf'{project_root}/Data/Azure/Combined_AzureLLMInferenceTrace_2023_Coding_2024_Conversation_compressed-5s.csv',

                        '24_Code_24_Conv_5s': rf'{project_root}/Data/Azure/Combined_AzureLLMInferenceTrace_2024_Coding_2024_Conversation_compressed-5s.csv',

            },
            #
            'TEST_TASK_SET':{'DC_LOAD_SCHEDULING':{
                                                    'ETA': 0.7,
                                                    'A':20,
                                                    'B':1.5,
                                                    'CAPABILITY_RANGE': np.array([0, 200, 400000]),
                                                    'POWER_RANGE': np.array([100, 280, 170]),
                                                   }
                            },

                            'TRAIN_DIFFUSION': False,
                            'GENERATE_FAKE_DATA': False,
                            'EVALUATE_FAKE_DATA': False,
                            'TRAIN_ML': False,
                            'TRAIN_DAGDRO': False,
                            'TRAIN_KLDRO': False,
                            'TRAIN_FWDRO': False,
                            'TEST_METHODS': True,
                            }

dfu_params = {

            'PUBLIC': {
                        'NAME':diffusion_name,
                        'SELECTED_EPOCH':7000,
                       },


            'TRAIN':{
                        'T': 500,
                        'BETA_1': 5e-2,
                        'BETA_T': 1e-1,
                        'BATCH_SIZE': 64,
                        'LR':1e-4,
                        'ITERATION':7000,
                        'SAVE_EVERY': 1000,
                        },


            'GENERATE': {
                        'T':500,
                        'BATCH_REPEAT':4,
                        'BETA_1': 5e-2,
                        'BETA_T': 1e-1,
                        'BATCH_SIZE': 64,
                        'SAVE_TRAJECTORY': True
                        },
                }


ml_params = {

             'PUBLIC': {
                        'NAME':ml_name,
                        'WINDOW_LEN': 7,
                        'PREDICT_LEN':1,
                        'SELECTED_EPOCH':100,
                        'DISPLAY_EVERY': 5,
                        'SAVE_EVERY': 10,

                        },


             'TRAIN':{
                         'ITERATION':100,
                         'DISCOUNT_FACTOR':0.05,
                         'STEP_SIZE':10,
                         'LR':1e-6,
                         'BATCH_SIZE':64,
             },


}

dag_params = {
              'PUBLIC': {
                        'NAME':dag_name,
                        'BATCH_SIZE': 64,
                        'DISPLAY_EVERY': 1,
                        'DISCOUNT_FACTOR':0.05,
                        'STEP_SIZE':2,
                        'PPO_CLIP':0.4
                    },
              'DRO':{
                        'ITERATION':15,
                        'ETA':0.01,
                        'MU': 1,
                        'BUDGET': 0.030,
                        'LR':1e-6,
                        'ML_SAVE_EVERY': 2,
                        'DFU_SAVE_EVERY': 2,
                        'ADJUST_TIMESTEPS': 10,
                        'P_S0':0.3

                    },
              'DIFFUSION':{
                        'PUBLIC':{
                                    'NAME':diffusion_name,
                                    'BETA_1': 5e-2,
                                    'BETA_T': 1e-1,
                                    'T': 500,
                                },

                        'TRAIN': {
                                    'BATCH_REPEAT': 4,
                                    'ITERATION':10,

                                },


                        'GENERATE': {
                                    }
                            },
              'ML':{
                        'PUBLIC':{
                                    'WINDOW_LEN': 7,
                                    'PREDICT_LEN':1,

                                },
                        'TRAIN':{
                                    'LR': 1e-6,
                                    'ITERATION':1,
                                }
              }
             }


fw_params = {
            'PUBLIC': {
                        'NAME':fw_name,
                        'WINDOW_LEN': 7,
                        'PREDICT_LEN':1,
                        'SELECTED_EPOCH':100,
                        'BATCH_SIZE':64,
                        'DISPLAY_EVERY': 5,
                        'SAVE_EVERY': 10,
                        },

            'DRO': {
                     'ITERATION':100,
                     'DISCOUNT_FACTOR':0.05,
                     'STEP_SIZE':10,
                     'BUDGET':2,
                     'ATTACK_STEPS':15,
                     'P':2,
                     'Q':2,
                     'LR':2e-5,
                    },
            }


kl_params = {
            'PUBLIC': {
                        'NAME':kl_name,
                        'WINDOW_LEN': 7,
                        'PREDICT_LEN':1,
                        'SELECTED_EPOCH':100,
                        'BATCH_SIZE':64,
                        'DISPLAY_EVERY': 5,
                        'SAVE_EVERY': 10,
                        },

            'DRO': {
                     'ITERATION':100,
                     'DISCOUNT_FACTOR':0.05,
                     'STEP_SIZE':10,
                     'BUDGET':2,
                     'ALPHA':1,
                     'LR':2e-5,
                    },
            }

log_folder_path,log_file_path = Log_Redirect().build_log_file(result_path=shared_params['RESULT_SAVE'],cuda=shared_params['DEVICE'])


sys.stdout = Tee(log_file_path)

with open(log_file_path, 'a') as f:
    print(f'  | Log Path |  {log_file_path}')

    dc_task_set_param = shared_params['TEST_TASK_SET']['DC_LOAD_SCHEDULING']
    dcs = DC_Scheduling(eta=dc_task_set_param['ETA'],
                        a=dc_task_set_param['A'],
                        b=dc_task_set_param['B'],
                        capability_range=dc_task_set_param['CAPABILITY_RANGE'],
                        power_range=dc_task_set_param['POWER_RANGE'])


    dfu_train_params = dfu_params['TRAIN']
    dfu_public_params = dfu_params['PUBLIC']
    dfu_generate_params = dfu_params['GENERATE']

    '''
    Read Training Dataset
    '''
    train_data_df = pd.read_csv(shared_params['TRAIN_FILE'], parse_dates=["TIMESTAMP"], date_format="mixed", dayfirst=False)['ContextTokens']


    '''
    Diffusion Pre-train Process
    '''

    if shared_params['TRAIN_DIFFUSION']:

        dfu_train = Train_DFU(
                            T=dfu_train_params['T'],
                            beta_1=dfu_train_params['BETA_1'],
                            beta_T=dfu_train_params['BETA_T'],
                            batch_size=dfu_train_params['BATCH_SIZE'],
                            lr=dfu_train_params['LR'],
                            total_iteration=dfu_train_params['ITERATION'],
                            picture_size=shared_params['PIC_SIZE'],
                            device=shared_params['DEVICE'],
                            data_df=train_data_df,
                            save_every=dfu_train_params['SAVE_EVERY'],
                            model_save_folder_path=shared_params['MODEL_SAVE'],
                            )

        dfu_train.train()


    '''
    Selected z0 Data
    '''
    dfu_model_path = cfg.diffusion_model_path(
                                            folder=shared_params['MODEL_SAVE'],
                                            model_name=dfu_public_params['NAME'],
                                            cuda=shared_params['DEVICE'],
                                            epoch=dfu_public_params['SELECTED_EPOCH'],
                                            batch_size=dfu_train_params['BATCH_SIZE'],
                                            pic_size=shared_params['PIC_SIZE'],
                                            T=dfu_train_params['T'],
                                            lr=dfu_train_params['LR'],
                                            beta_1=dfu_train_params['BETA_1'],
                                            beta_T=dfu_train_params['BETA_T']
                                            )

    z0_data_path = cfg.generate_data_path(
                                            folder=shared_params['GEN_DATA_SAVE'],
                                            model_name=dfu_public_params['NAME'],
                                            cuda=shared_params['DEVICE'],
                                            epoch=dfu_public_params['SELECTED_EPOCH'],
                                            batch_size=dfu_generate_params['BATCH_SIZE'],
                                            pic_size=shared_params['PIC_SIZE'],
                                            T=dfu_generate_params['T'],
                                            beta_1=dfu_generate_params['BETA_1'],
                                            beta_T=dfu_generate_params['BETA_T'],
                                            )

    z0_trajectory_path = cfg.trajectory_data_path(
                                            folder=shared_params['GEN_DATA_SAVE'],
                                            model_name=dfu_public_params['NAME'],
                                            cuda=shared_params['DEVICE'],
                                            epoch=dfu_public_params['SELECTED_EPOCH'],
                                            batch_size=dfu_generate_params['BATCH_SIZE'],
                                            pic_size=shared_params['PIC_SIZE'],
                                            T=dfu_generate_params['T'],
                                            beta_1=dfu_generate_params['BETA_1'],
                                            beta_T=dfu_generate_params['BETA_T'],
                                            )




    '''
    Generate z0 via Diffusion
    '''
    if shared_params['GENERATE_FAKE_DATA']:
        test_data_df = train_data_df

        inference = Inference_DFU(
                              T=dfu_generate_params['T'],
                              beta_1=dfu_generate_params['BETA_1'],
                              beta_T=dfu_generate_params['BETA_T'],
                              batch_size=dfu_generate_params['BATCH_SIZE'],
                              device=shared_params['DEVICE'],
                              picture_size=shared_params['PIC_SIZE'],
                              sampling_number=dfu_generate_params['BATCH_SIZE'],

                              )
    #
        inference.generate_wholeset_data(
                                         model_path=dfu_model_path,
                                         batch_repeat=dfu_generate_params['BATCH_REPEAT'],
                                         data_df=test_data_df,
                                         save_trajectory=dfu_generate_params['SAVE_TRAJECTORY'],
                                         save_gen_data_path=z0_data_path,
                                         )

    '''
    Evaluate z0
    '''
    if shared_params['EVALUATE_FAKE_DATA']:
        print(f'\n=============== Evaluating Generate Data Start ===============')
        print(f'  | Selected Epoch | {dfu_public_params['SELECTED_EPOCH']}')
        print(f'  | File Name | {z0_data_path}')

        z0_data_df = pd.read_csv(z0_data_path)['ContextTokens']

        Evaluation().fid_evaluation(
                                    fake_df=z0_data_df,
                                    true_df=train_data_df,
                                    picture_size=shared_params['PIC_SIZE'],
                                    device=shared_params['DEVICE'],
                                    isshow=False
                                    )
        dist = Utils().compute_normalized_wasserstein(z0_data_df, train_data_df)
        print(f"  | Wasserstein Distance | {dist}")
        print(f'=============== Evaluating Generate Data End ===============')


    '''
    ML Pre-train Process
    '''
    ml_train_params = ml_params['TRAIN']
    ml_public_params = ml_params['PUBLIC']

    ml = ML(
            windows_len=ml_public_params['WINDOW_LEN'],
            predict_len=ml_public_params['PREDICT_LEN'],
            epoches=ml_train_params['ITERATION'],
            discount_factor=ml_train_params['DISCOUNT_FACTOR'],
            step_size=ml_train_params['STEP_SIZE'],
            lr=ml_train_params['LR'],
            device=shared_params['DEVICE'],
            display_every =ml_public_params['DISPLAY_EVERY'],
            save_every = ml_public_params['SAVE_EVERY']
            )

    ml_model_path = cfg.ml_model_path(
                                        folder=shared_params['MODEL_SAVE'],
                                        model_name=ml_public_params['NAME'],
                                        cuda=shared_params['DEVICE'],
                                        epoch=ml_public_params['SELECTED_EPOCH'],
                                        lr=ml_train_params['LR'],
                                      )



    if shared_params['TRAIN_ML']:
        ml_dl = Data_Loader(df=train_data_df)
        # If you want the training set without Noise, then do: noise_type=None. It will ignore noise_param as well. You can also let:
        # noise_type='cutout' or 'gaussian' or 'perlin' to introduce different kind of noise in the training set with different noise_param.
        ml_train_loader, ml_test_loader = ml_dl.ml_set2loader(
            split_test=True,
            normlized=True,
            device=shared_params['DEVICE'],
            batch_size=ml_train_params['BATCH_SIZE'],
            pic_size=shared_params['PIC_SIZE'],
            noise_type=None,

            noise_param=0.05
        )

        # If you want to train the model with MSE, then do: task=None. If you want to train with the decision focused learning task, please let task = dcs and data_range=Data_Loader(df=train_data_df).data_group.
        #  ml.train(
        #      model_path=None,
        #      model_save_folder_path=shared_params['MODEL_SAVE'],
        #      train_data_loader=ml_train_loader,
        #  )

        ml.train(
                    task = dcs,
                    model_path= None,
                    model_save_folder_path=shared_params['MODEL_SAVE'],
                    train_data_loader=ml_train_loader,
                    data_range=Data_Loader(df=train_data_df).data_group
                )


        Utils().draw_loss_plot(
            loss_list=ml.epoches_losses_list,
            log_folder_path=log_folder_path,
            ml_name=ml_name,
        )

        # If you want to test within MSE, please use the follow test function

        # ml.test(
        #         model_path=ml_model_path,
        #         test_data_loader=ml_test_loader,
        # )

        # If you want to test within the task, please use the follow test_task function
        ml.test_task(
            model_path=ml_model_path,
            test_data_loader=ml_test_loader,
            task=dcs,
            data_range=Data_Loader(df=train_data_df).data_group

        )


    '''
    Load Z0 & trajectory
    '''
    z0_data_df = pd.read_csv(z0_data_path)['ContextTokens']
    z0_trajectory = torch.tensor(np.load(z0_trajectory_path), dtype=torch.float32, device=shared_params['DEVICE'])
    z0_timesteps = list(range(0, dfu_generate_params['T'], 1))




    '''
    DAG-DRO Train Process
    '''
    dag_public_params = dag_params['PUBLIC']
    dag_dro_params = dag_params['DRO']
    dag_dfu_params = dag_params['DIFFUSION']
    dag_ml_params = dag_params['ML']


    dag = DAG_DRO(
                    beta_1=dag_dfu_params['PUBLIC']['BETA_1'],
                    beta_T=dag_dfu_params['PUBLIC']['BETA_T'],
                    T=dag_dfu_params['PUBLIC']['T'],
                    ppo_clip=dag_public_params['PPO_CLIP'],
                    adjust_timesteps=dag_dro_params['ADJUST_TIMESTEPS'],
                    batch_repeat=dag_dfu_params['TRAIN']['BATCH_REPEAT'],
                    batch_size=dag_public_params['BATCH_SIZE'],
                    pic_size=shared_params['PIC_SIZE'],
                    discount_factor=dag_public_params['DISCOUNT_FACTOR'],
                    step_size=dag_public_params['STEP_SIZE'],
                    dro_lr=dag_dro_params['LR'],
                    epoches=dag_dro_params['ITERATION'],
                    dro_inner_epochs = dag_dfu_params['TRAIN']['ITERATION'],
                    ml_inner_epochs = dag_ml_params['TRAIN']['ITERATION'],
                    p_s0=dag_dro_params['P_S0'],
                    eta=dag_dro_params['ETA'],
                    mu=dag_dro_params['MU'],
                    budget=dag_dro_params['BUDGET'],
                    windows_len=dag_ml_params['PUBLIC']['WINDOW_LEN'],
                    predict_len=dag_ml_params['PUBLIC']['PREDICT_LEN'],
                    display_every=dag_public_params['DISPLAY_EVERY'],
                    ml_lr=dag_ml_params['TRAIN']['LR'],
                    selected_timesteps=z0_timesteps,
                    ml_save_every=dag_dro_params['ML_SAVE_EVERY'],
                    dfu_save_every=dag_dro_params['DFU_SAVE_EVERY'],
                    dfu_name=dag_dfu_params['PUBLIC']['NAME'],
                    device=shared_params['DEVICE'],
                    )

    dag_dro_selected_epoch = (dag_dro_params['ITERATION'] // dag_dro_params['ML_SAVE_EVERY']) * \
                             dag_dro_params['ML_SAVE_EVERY']

    selected_dag_dro_ml_model_path = cfg.dro_save_model_path(
        folder=shared_params['MODEL_SAVE'],
        dro_name=dag_name,
        model_name=ml_name,
        cuda=shared_params['DEVICE'],
        epoch=dag_dro_selected_epoch,
        lr=dag_ml_params['TRAIN']['LR']
    )

    if shared_params['TRAIN_DAGDRO']:
        # If you want to train the model with MSE, then do: task=None. If you want to train with the decision focused learning task, please let task = dcs and data_range=Data_Loader(df=train_data_df).data_group.
        # dag.train(
        #     ml_model_path=ml_model_path,
        #     dfu_model_path=dfu_model_path,
        #     z0=z0_data_df,
        #     s0=train_data_df,
        #     z0_trajectory=z0_trajectory,
        #     model_save_folder_path=shared_params['MODEL_SAVE'],
        # )

        dag.train(
                    ml_model_path=ml_model_path,
                    dfu_model_path=dfu_model_path,
                    z0=z0_data_df,
                    s0=train_data_df,
                    z0_trajectory=z0_trajectory,
                    model_save_folder_path = shared_params['MODEL_SAVE'],
                    task=dcs,
                    data_range=Data_Loader(df=train_data_df).data_group
                 )


        Utils().draw_loss_plot(
                               loss_list=dag.epoches_losses_list,
                               log_folder_path=log_folder_path,
                               dro_name=dag_name,
                               diffusion_name=diffusion_name,
                               ml_name=ml_name,
                                )


    '''
    FW-DRO Train Process
    '''
    fw_public_params = fw_params['PUBLIC']
    fw_dro_params = fw_params['DRO']

    fw = FW_DRO(
                windows_len=fw_public_params['WINDOW_LEN'],
                predict_len=fw_public_params['PREDICT_LEN'],
                epoches=fw_dro_params['ITERATION'],
                discount_factor=fw_dro_params['DISCOUNT_FACTOR'],
                step_size=fw_dro_params['STEP_SIZE'],
                budget=fw_dro_params['BUDGET'],
                attack_steps=fw_dro_params['ATTACK_STEPS'],
                p=fw_dro_params['P'],
                q=fw_dro_params['Q'],
                lr=fw_dro_params['LR'],
                device=shared_params['DEVICE'],
                save_every=fw_public_params['SAVE_EVERY'],
                display_every=fw_public_params['DISPLAY_EVERY'],
                )

    fw_dro_model_path = cfg.dro_save_model_path(
                    folder=shared_params['MODEL_SAVE'],
                    dro_name=fw_public_params['NAME'],
                    model_name='ML',
                    cuda=shared_params['DEVICE'],
                    epoch=fw_public_params['SELECTED_EPOCH'],
                    lr=fw_dro_params['LR'],
                    )

    if shared_params['TRAIN_FWDRO']:
        fw_dro_dl = Data_Loader(df=train_data_df)
        fw_dro_train_loader, fw_dro_test_loader = fw_dro_dl.ml_set2loader(
            split_test=True,
            normlized=True,
            device=shared_params['DEVICE'],
            batch_size=fw_public_params['BATCH_SIZE'],
            pic_size=shared_params['PIC_SIZE'],
        )

        # If you want to train the model with MSE, then do: task=None. If you want to train with the decision focused learning task, please let task = dcs and data_range=Data_Loader(df=train_data_df).data_group.
        # fw.train(
        #     model_path=ml_model_path,
        #     model_save_folder_path=shared_params['MODEL_SAVE'],
        #     train_data_loader=fw_dro_train_loader,
        #
        # )

        fw.train(
            model_path=ml_model_path,
            model_save_folder_path=shared_params['MODEL_SAVE'],
            train_data_loader=fw_dro_train_loader,
            task=dcs,
            data_range=Data_Loader(df=train_data_df).data_group
        )

        Utils().draw_loss_plot(
            loss_list=fw.epoches_losses_list,
            log_folder_path=log_folder_path,
            dro_name=fw_name,
        )

        # If you want to test within MSE, please use the follow test function

        # fw.test(
        #         model_path=fw_dro_model_path,
        #         test_data_loader=fw_dro_test_loader,
        #     )

        # If you want to test within the task, please use the follow test_task function

        fw.test_task(
            model_path=fw_dro_model_path,
            test_data_loader=fw_dro_test_loader,
            task=dcs,
            data_range=Data_Loader(df=train_data_df).data_group

        )

    '''
    KL-DRO Train Process
    '''
    kl_public_params = kl_params['PUBLIC']
    kl_dro_params = kl_params['DRO']

    kl = KL_DRO(
                windows_len=kl_public_params['WINDOW_LEN'],
                predict_len=kl_public_params['PREDICT_LEN'],
                epoches=kl_dro_params['ITERATION'],
                discount_factor=kl_dro_params['DISCOUNT_FACTOR'],
                step_size=kl_dro_params['STEP_SIZE'],
                alpha=kl_dro_params['ALPHA'],
                budget=kl_dro_params['BUDGET'],
                lr=kl_dro_params['LR'],
                device=shared_params['DEVICE'],
                save_every=kl_public_params['SAVE_EVERY'],
                display_every=kl_public_params['DISPLAY_EVERY'],
            )

    kl_dro_model_path = cfg.dro_save_model_path(
        folder=shared_params['MODEL_SAVE'],
        dro_name=kl_public_params['NAME'],
        model_name='ML',
        cuda=shared_params['DEVICE'],
        epoch=kl_public_params['SELECTED_EPOCH'],
        lr=kl_dro_params['LR'],
    )



    if shared_params['TRAIN_KLDRO']:
        kl_dro_dl = Data_Loader(df=train_data_df)
        kl_dro_train_loader, kl_dro_test_loader = kl_dro_dl.ml_set2loader(
            split_test=True,
            normlized=True,
            device=shared_params['DEVICE'],
            batch_size=kl_public_params['BATCH_SIZE'],
            pic_size=shared_params['PIC_SIZE'],
        )

        # If you want to train the model with MSE, then do: task=None. If you want to train with the decision focused learning task, please let task = dcs and data_range=Data_Loader(df=train_data_df).data_group.
        # kl.train(
        #     model_path=ml_model_path,
        #     model_save_folder_path=shared_params['MODEL_SAVE'],
        #     train_data_loader=kl_dro_train_loader,
        # )

        kl.train(
            model_path=ml_model_path,
            model_save_folder_path=shared_params['MODEL_SAVE'],
            train_data_loader=kl_dro_train_loader,
            task=dcs,
            data_range=Data_Loader(df=train_data_df).data_group
        )

        Utils().draw_loss_plot(
            loss_list=kl.epoches_losses_list,
            log_folder_path=log_folder_path,
            dro_name=kl_name,
        )

        # If you want to test within MSE, please use the follow test function

        # kl.test(
        #     model_path=kl_dro_model_path,
        #     test_data_loader=kl_dro_test_loader,
        # )

        # If you want to test within the task, please use the follow test_task function

        kl.test_task(
            model_path=kl_dro_model_path,
            test_data_loader=kl_dro_test_loader,
            task=dcs,
            data_range=Data_Loader(df=train_data_df).data_group

        )



    if shared_params['TEST_METHODS']:

        print(f'\n=============== Testing Methods Start ===============')
        for test_set_name,test_set_path in shared_params['TEST_SET'].items():
            print('\033[92m\n *****************************************************\n\033[0m')
            print(f'  | Test Dataset | \033[93m{test_set_name}\033[0m')
            print(f'  | Dateset Path | {test_set_path}')

            try:
                eva_data_df = pd.read_csv(test_set_path, parse_dates=["TIMESTAMP"], date_format="mixed", dayfirst=False)['ContextTokens']
            except:
                eva_data_df = pd.read_csv(test_set_path)['ContextTokens']


            if test_set_path==shared_params['TRAIN_FILE']:
                split_test = True

            else:
                split_test = False

            # Read Data
            eva_dl = Data_Loader(df=eva_data_df,len_limit=100000)

            loader_output = eva_dl.ml_set2loader(
                    split_test=split_test,
                    normlized=True,
                    device=shared_params['DEVICE'],
                    batch_size=ml_train_params['BATCH_SIZE'],
                    pic_size=shared_params['PIC_SIZE'],
                )

            # If you want to inject different type of noise, please open the follow comment.

            ## prelin -0.05
            ## cutout - 0.005
            ## gaussian - 0.1

            # loader_output = eva_dl.ml_set2loader(
            #     split_test=split_test,
            #     normlized=True,
            #     device=shared_params['DEVICE'],
            #     batch_size=ml_train_params['BATCH_SIZE'],
            #     pic_size=shared_params['PIC_SIZE'],
            #     noise_type = 'gaussian',
            #     noise_param = 0.1
            # )

            eva_loader = loader_output[1] if split_test else loader_output

            # Evaluate testset on ML
            # ml.test(
            #         model_path=ml_model_path,
            #         test_data_loader=eva_loader,
            #     )

            ml.test_task(
                    model_path=ml_model_path,
                    test_data_loader=eva_loader,
                    task=dcs,
                    data_range = Data_Loader(df=train_data_df).data_group
                )

            # # #  Evaluate testset on FW DRO
            # # fw.test(
            # #         model_path=fw_dro_model_path,
            # #         test_data_loader=eva_loader,
            # #     )
            #
            fw.test_task(
                model_path=fw_dro_model_path,
                test_data_loader=eva_loader,
                task=dcs,
                data_range=Data_Loader(df=train_data_df).data_group
            )
            #
            # # #  Evaluate testset on KL DRO
            # # kl.test(
            # #         model_path=kl_dro_model_path,
            # #         test_data_loader=eva_loader,
            # #     )
            #
            kl.test_task(
                model_path=kl_dro_model_path,
                test_data_loader=eva_loader,
                task=dcs,
                data_range=Data_Loader(df=train_data_df).data_group
                )

            #
            # #  Evaluate testset on DAG DRO
            # dag.test(
            #         model_path=selected_dag_dro_ml_model_path,
            #         test_data_loader=eva_loader,
            #     )

            dag.test_task(
                model_path=selected_dag_dro_ml_model_path,
                test_data_loader=eva_loader,
                task=dcs,
                data_range=Data_Loader(df=train_data_df).data_group
            )

        print('\033[92m\n *****************************************************\n\033[0m')
        print(f'=============== Testing Methods End ===============')


# === Restore Output ===
print(f"\n  | Log Saved | {log_file_path}")
sys.stdout = sys.__stdout__

