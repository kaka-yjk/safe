import os
import logging
import pickle
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader

__all__ = ['MMDataLoader']

logger = logging.getLogger('MSA')


def load_all_rationales(args, modality):
    master_rationale_dict = {}
    for split in ['train', 'valid', 'test']:
        path = getattr(args, f'rationale_{modality}_path', '').replace('train', split)
        if path and os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    master_rationale_dict.update(data)
                logger.info(f"Successfully loaded {len(data)} rationales from {path}")
            except Exception as e:
                logger.error(f"Error loading rationale file {path}: {e}")
        else:
            logger.warning(f"Rationale file not found for modality '{modality}' in split '{split}': {path}")

    return master_rationale_dict


class MMDataset(Dataset):
    def __init__(self, args, mode='train', all_rationales=None):
        self.mode = mode
        self.args = args

        if all_rationales:
            self.rationales_text = all_rationales.get('text', {})
            self.rationales_vision = all_rationales.get('vision', {})
            self.rationales_audio = all_rationales.get('audio', {})
        else:
            logger.warning("No pre-loaded master rationale dictionary found.")
            self.rationales_text, self.rationales_vision, self.rationales_audio = {}, {}, {}

        with open(self.args.dataPath, 'rb') as f:
            data = pickle.load(f)[self.mode]

        self.vision = data['vision'].astype(np.float32)
        self.audio = data['audio'].astype(np.float32)
        self.text = data['text_bert'].astype(np.float32)
        self.labels = data[self.args.train_mode + '_labels'].astype(np.float32)

        self.pkl_ids = data['id']

        self.aligned_keys = []
        self.aligned_indices = []

        for i, pkl_id in enumerate(self.pkl_ids):
            sample_key = ""
            dataset_name = self.args.datasetName.lower()

            if dataset_name == 'sims':
                try:
                    parts = pkl_id.split('$_$')
                    video_part = parts[0]
                    segment_part = str(int(parts[1]))
                    sample_key = f"{video_part}_{segment_part}"
                except (IndexError, ValueError):
                    sample_key = pkl_id

            else:
                sample_key = pkl_id.replace('$_$', '_')

            if sample_key in self.rationales_text and \
                    sample_key in self.rationales_vision and \
                    sample_key in self.rationales_audio:
                self.aligned_keys.append(sample_key)
                self.aligned_indices.append(i)

        logger.info(f"[{mode.upper()} SET] Originally found {len(self.pkl_ids)} samples in pkl.")
        logger.info(
            f"[{mode.upper()} SET] After alignment with master rationale DB, {len(self.aligned_keys)} samples are available.")

        if len(self.aligned_keys) == 0 and len(self.pkl_ids) > 0:
            logger.error(
                f"[{mode.upper()} SET] Alignment failed! No common IDs found. Please check the ID conversion logic in `data/load_data.py` for the '{dataset_name}' dataset.")
            logger.error(
                f"Example pkl ID: '{self.pkl_ids[0]}'. Example JSON key from user: 'video_0001_1' or '03bSnISJMiM_11'.")

    def __len__(self):
        return len(self.aligned_keys)

    def __getitem__(self, index):
        sample_key = self.aligned_keys[index]
        original_index = self.aligned_indices[index]

        return {
            'text': torch.tensor(self.text[original_index]),
            'audio': torch.tensor(self.audio[original_index]),
            'vision': torch.tensor(self.vision[original_index]),
            'labels': torch.tensor(self.labels[original_index].reshape(-1)),
            'rationale_text': self.rationales_text[sample_key],
            'rationale_vision': self.rationales_vision[sample_key],
            'rationale_audio': self.rationales_audio[sample_key]
        }


def MMDataLoader(args):
    all_rationales_db = {
        'text': load_all_rationales(args, 'text'),
        'vision': load_all_rationales(args, 'vision'),
        'audio': load_all_rationales(args, 'audio'),
    }

    datasets = {
        mode: MMDataset(args, mode=mode, all_rationales=all_rationales_db)
        for mode in ['train', 'valid', 'test']
    }

    def collate_fn(batch):
        elem = batch[0]
        res = {key: [d[key] for d in batch] for key in elem if
               key not in ['rationale_text', 'rationale_vision', 'rationale_audio']}

        for key in res:
            if isinstance(res[key][0], torch.Tensor):
                res[key] = torch.stack(res[key])

        res['rationale_text'] = [d['rationale_text'] for d in batch]
        res['rationale_vision'] = [d['rationale_vision'] for d in batch]
        res['rationale_audio'] = [d['rationale_audio'] for d in batch]

        return res

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       shuffle=(ds == 'train'),
                       collate_fn=collate_fn)
        for ds in datasets.keys()
    }

    return dataLoader