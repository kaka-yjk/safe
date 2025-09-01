import os
import argparse
from utils.functions import Storage


class ConfigRegression():
    def __init__(self, args):
        HYPER_MODEL_MAP = {
            'safe': self.__SAFE
        }
        HYPER_DATASET_MAP = self.__datasetCommonParams()

        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)

        if model_name not in HYPER_MODEL_MAP:
            raise ValueError(f"Unknown model: {args.modelName}. Please check your --modelName argument.")

        if dataset_name not in HYPER_DATASET_MAP:
            raise ValueError(f"Unknown dataset: {args.datasetName}. Please check your --datasetName argument.")

        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]
        dataArgs = dataArgs['aligned'] if (commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs[
            'unaligned']

        self.args = Storage(dict(vars(args),
                                 **dataArgs,
                                 **commonArgs,
                                 **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                                 ))

    def __datasetCommonParams(self):
        root_dataset_dir = '/home/kaka/data'
        root_rationale_dir = '/home/kaka/oldrationale/json_fixed'
        return {
            'mosi': {
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/unaligned_50.pkl'),
                    'rationale_text_path': os.path.join(root_rationale_dir, 'rationales_train_MOSI_text.json'),
                    'rationale_vision_path': os.path.join(root_rationale_dir, 'rationales_train_MOSI_vision.json'),
                    'rationale_audio_path': os.path.join(root_rationale_dir, 'rationales_train_MOSI_audio.json'),
                    'seq_lens': (50, 500, 500),
                    'feature_dims': (768, 20, 5),
                    'train_samples': 1284,
                    'num_classes': 1,
                    'language': 'en',
                    'KeyEval': 'MAE'
                }
            },
            'sims': {
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'SIMS/Processed/unaligned_39.pkl'),
                    'rationale_text_path': os.path.join(root_rationale_dir, 'rationales_train_SIMS_text.json'),
                    'rationale_vision_path': os.path.join(root_rationale_dir, 'rationales_train_SIMS_vision.json'),
                    'rationale_audio_path': os.path.join(root_rationale_dir, 'rationales_train_SIMS_audio.json'),
                    'seq_lens': (39, 55, 400),
                    'feature_dims': (768, 709, 33),
                    'train_samples': 1368,
                    'num_classes': 1,
                    'language': 'cn',
                    'KeyEval': 'MAE'
                }
            }
        }

    def __SAFE(self):
        return {
            'commonParas': {
                'need_data_aligned': False,
                'need_model_aligned': False,
                'use_bert': True,
                'use_finetune': True,
            },
            'datasetParas': {
                'mosi': {
                    'bert_path': '/home/kaka/llms/bert-base-uncased',
                    'text_dim': 768,
                    'vision_dim': 20,
                    'audio_dim': 5,
                    'hidden_dim': 128,
                    'rsa_hidden_dim': 128,
                    'rsa_num_layers': 2,
                    'rsa_num_heads': 4,
                    'post_fusion_dropout': 0.2,
                    'a_encoder_heads': 1,
                    'v_encoder_heads': 4,
                    'a_encoder_layers': 2,
                    'v_encoder_layers': 2,
                    'rationale_max_len': 100,

                    'num_epochs': 100,
                    'batch_size': 64,
                    'gradient_accumulation_steps': 1,
                    'early_stop': 20,

                    'ppo_epochs': 4,
                    'clip_param': 0.1,

                    'entropy_coefficient': 0.012,
                    'value_loss_coefficient': 0.5,

                    'warm_up_epochs': 10,
                    'rl_start_epoch': 10,

                    'learning_rate_ppo': 1e-5,
                    'weight_decay_ppo': 0.01,

                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 1e-3,
                    'learning_rate_video': 5e-4,
                    'learning_rate_other': 2e-4,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.01,
                    'weight_decay_video': 0.01,
                    'weight_decay_other': 0.001,

                    'w_task': 1,
                    'w_parsimony': 0.01,

                    'action_threshold': 0.7
                },
                'sims': {
                    'bert_path': '/home/kaka/llms/bert-base-chinese',
                    'text_dim': 768,
                    'vision_dim': 709,
                    'audio_dim': 33,
                    'hidden_dim': 128,
                    'rsa_hidden_dim': 128,
                    'rsa_num_layers': 2,
                    'rsa_num_heads': 4,
                    'post_fusion_dropout': 0.2,
                    'a_encoder_heads': 3,
                    'v_encoder_heads': 1,
                    'a_encoder_layers': 2,
                    'v_encoder_layers': 2,
                    'rationale_max_len': 200,

                    'num_epochs': 100,
                    'batch_size': 64,
                    'gradient_accumulation_steps': 1,
                    'early_stop': 20,

                    'ppo_epochs': 4,
                    'clip_param': 0.1,

                    'entropy_coefficient': 0.012,
                    'value_loss_coefficient': 0.5,

                    'warm_up_epochs': 10,
                    'rl_start_epoch': 10,

                    'learning_rate_ppo': 1e-5,
                    'weight_decay_ppo': 0.01,

                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 5e-5,
                    'learning_rate_video': 1.2e-5,
                    'learning_rate_other': 1.5e-4,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.01,
                    'weight_decay_video': 0.01,
                    'weight_decay_other': 0.001,

                    'w_task': 1,
                    'w_parsimony': 0.01,

                    'action_threshold': 0.7
                }
            },
        }

    def get_config(self):
        return self.args