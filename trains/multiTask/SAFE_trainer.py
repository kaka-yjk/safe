import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import logging
import numpy as np
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.metricsTop import MetricsTop
from utils.functions import dict_to_str
import matplotlib.pyplot as plt
import os

logger = logging.getLogger('MSA')


class PPOBuffer:
    def __init__(self):
        self.states, self.actions, self.logprobs, self.rewards, self.state_values, self.is_terminals = [], [], [], [], [], []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class SAFE_trainer:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)
        self.buffer = PPOBuffer()

        self.policy = model.Model.drs_policy

        bert_params, audio_params, video_params, other_params = [], [], [], []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'drs_policy' in name:
                continue
            if 'text_encoder' in name or 'rationale_encoder' in name:
                bert_params.append(param)
            elif 'audio_encoder' in name:
                audio_params.append(param)
            elif 'video_encoder' in name:
                video_params.append(param)
            else:
                other_params.append(param)

        optimizer_grouped_parameters = [
            {'params': bert_params, 'lr': args.learning_rate_bert, 'weight_decay': args.weight_decay_bert},
            {'params': audio_params, 'lr': args.learning_rate_audio, 'weight_decay': args.weight_decay_audio},
            {'params': video_params, 'lr': args.learning_rate_video, 'weight_decay': args.weight_decay_video},
            {'params': other_params, 'lr': args.learning_rate_other, 'weight_decay': args.weight_decay_other}
        ]

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=args.learning_rate_ppo,
            weight_decay=args.get('weight_decay_ppo', 0.01)
        )
        self.sup_optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

        self.TaskLoss = nn.L1Loss()
        self.MseLoss = nn.MSELoss()

    def _update_policy(self):
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(self.args.device)
        old_states = torch.stack(self.buffer.states, dim=0).detach().to(self.args.device)
        old_actions = torch.stack(self.buffer.actions, dim=0).detach().to(self.args.device)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().to(self.args.device)
        old_state_values = torch.tensor(self.buffer.state_values, dtype=torch.float32).to(self.args.device)

        advantages = rewards - old_state_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = rewards

        for _ in range(self.args.ppo_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.squeeze())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.args.clip_param, 1 + self.args.clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.MseLoss(state_values.squeeze(), returns)
            entropy_loss = dist_entropy.mean()
            loss = actor_loss + self.args.value_loss_coefficient * critic_loss - self.args.entropy_coefficient * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        self.buffer.clear()

    def do_train(self, dataloader):
        num_training_steps = len(dataloader['train']) * self.args.num_epochs
        num_warmup_steps = len(dataloader['train']) * self.args.warm_up_epochs

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
        self.sup_scheduler = get_linear_schedule_with_warmup(
            self.sup_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )

        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['MAE', 'Loss'] else 'max'
        best_valid = float('inf') if min_or_max == 'min' else float('-inf')

        history = {
            'train_loss': [], 'val_mae': [], 'sup_lr': [],
            'ppo_lr': [], 'best_epoch_results': {}
        }

        accumulation_steps = self.args.get('gradient_accumulation_steps', 1)

        while epochs < self.args.num_epochs:
            epochs += 1
            self.model.train()
            epoch_train_loss = 0

            is_rl_active = epochs > self.args.rl_start_epoch
            if is_rl_active:
                logger.info(f"Epoch {epochs}: RL training is active.")
            else:
                logger.info(
                    f"Epoch {epochs}: Supervised warm-up phase (RL training starts after epoch {self.args.rl_start_epoch}).")

            self.sup_optimizer.zero_grad()

            with tqdm(dataloader['train'], desc=f"Epoch {epochs}/{self.args.num_epochs}") as td:
                for i, batch_data in enumerate(td):
                    text, audio, vision, labels = batch_data['text'].to(self.args.device), batch_data['audio'].to(
                        self.args.device), \
                        batch_data['vision'].to(self.args.device), batch_data['labels'].to(self.args.device)

                    prediction, state, action, action_logprob = self.model(
                        text, audio, vision, batch_data['rationale_text'],
                        batch_data['rationale_vision'], batch_data['rationale_audio'],
                        deterministic=False
                    )

                    sup_loss = self.TaskLoss(prediction.squeeze(), labels.squeeze())
                    epoch_train_loss += sup_loss.item()
                    loss = sup_loss / accumulation_steps
                    loss.backward()

                    if (i + 1) % accumulation_steps == 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        self.sup_optimizer.step()
                        self.sup_scheduler.step()
                        self.sup_optimizer.zero_grad()

                    if is_rl_active:
                        with torch.no_grad():
                            _, state_values = self.policy(state)
                            r_task = -torch.abs(prediction.squeeze() - labels.squeeze()).detach()

                            r_parsimony = -torch.sum(action, dim=1)

                            reward = self.args.w_task * r_task + self.args.w_parsimony * r_parsimony

                            self.buffer.states.extend(state.cpu())
                            self.buffer.actions.extend(action.cpu())
                            self.buffer.logprobs.extend(action_logprob.cpu())
                            self.buffer.rewards.extend(reward.cpu())
                            self.buffer.state_values.extend(state_values.squeeze().cpu())
                            self.buffer.is_terminals.extend([True] * len(reward))

            if is_rl_active:
                self._update_policy()
                self.scheduler.step()

            val_results = self.do_test(dataloader['valid'], mode="VAL")

            history['train_loss'].append(epoch_train_loss / len(dataloader['train']))
            history['val_mae'].append(val_results['MAE'])
            history['sup_lr'].append(self.sup_optimizer.param_groups[0]['lr'])
            history['ppo_lr'].append(self.optimizer.param_groups[0]['lr'])

            logger.info(f"VAL-{self.args.modelName} Epoch {epochs} >> {dict_to_str(val_results)}")

            isBetter = val_results[self.args.KeyEval] < best_valid if min_or_max == 'min' else val_results[
                                                                                                   self.args.KeyEval] > best_valid
            if isBetter:
                best_valid, best_epoch = val_results[self.args.KeyEval], epochs
                history['best_epoch_results'] = val_results
                history['best_epoch_results']['epoch'] = best_epoch
                torch.save(self.model.cpu().state_dict(), self.args.model_save_path)
                self.model.to(self.args.device)
                logger.info(f"Found new best model at epoch {epochs}, saving to {self.args.model_save_path}")

            if epochs - best_epoch >= self.args.early_stop:
                logger.info(f"Early stopping at epoch {epochs}")
                break

        self._generate_plots(history)
        return history['best_epoch_results']

    def do_test(self, dataloader, mode="TEST"):
        self.model.eval()
        y_pred, y_true = [], []
        with torch.no_grad(), tqdm(dataloader, desc=f"{mode} Evaluation") as td:
            for batch_data in td:
                text = batch_data['text'].to(self.args.device)
                audio = batch_data['audio'].to(self.args.device)
                vision = batch_data['vision'].to(self.args.device)
                labels = batch_data['labels'].to(self.args.device)

                prediction, _, _, _ = self.model(
                    text, audio, vision, batch_data['rationale_text'],
                    batch_data['rationale_vision'], batch_data['rationale_audio'],
                    deterministic=True
                )


                y_pred.append(prediction.cpu())
                y_true.append(labels.cpu())
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred.squeeze(), true.squeeze())
        eval_results['Loss'] = F.l1_loss(pred.squeeze(), true.squeeze()).item()
        logger.info(f"{mode} Metrics: >> {dict_to_str(eval_results)}")
        return eval_results

    def _generate_plots(self, history):
        plot_dir = 'results/plots'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        epochs = range(1, len(history['val_mae']) + 1)

        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training & Validation Metrics for seed {self.args.seed}', fontsize=16)

        axs[0, 0].plot(epochs, history['val_mae'], marker='o', linestyle='-', label='Validation MAE')
        axs[0, 0].set_title('Validation MAE Curve')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('MAE')
        axs[0, 0].grid(True)
        axs[0, 0].legend()

        axs[0, 1].plot(epochs, history['train_loss'], marker='o', linestyle='-', color='orange', label='Training Loss')
        axs[0, 1].set_title('Training Loss Curve')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].grid(True)
        axs[0, 1].legend()

        axs[1, 0].plot(epochs, history['sup_lr'], marker='.', linestyle='-', color='green', label='Supervised LR')
        axs[1, 0].plot(epochs, history['ppo_lr'], marker='.', linestyle='-', color='purple', label='PPO LR')
        axs[1, 0].set_title('Learning Rate Schedule')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Learning Rate')
        axs[1, 0].grid(True)
        axs[1, 0].legend()

        ax4 = axs[1, 1]
        ax4_twin = ax4.twinx()
        ax4.plot(epochs, history['train_loss'], marker='o', linestyle='-', color='orange', label='Training Loss')
        ax4_twin.plot(epochs, history['val_mae'], marker='o', linestyle='-', color='blue', label='Validation MAE')
        ax4.set_title('Training Loss vs. Validation MAE')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Training Loss', color='orange')
        ax4_twin.set_ylabel('Validation MAE', color='blue')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        ax4.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(plot_dir,
                                 f'{self.args.modelName}-{self.args.datasetName}-seed-{self.args.seed}-metrics.png')
        plt.savefig(save_path)
        plt.close()