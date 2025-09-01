import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from torch.distributions import Bernoulli
import math

from models.subNets.BertTextEncoder import BertTextEncoder


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, device, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(f"Input sequence length ({seq_len}) is greater than max_len ({self.pe.size(1)})")
        x = x + self.pe[:, :seq_len].requires_grad_(False)
        return self.dropout(x)


class AuVi_Encoder(nn.Module):
    def __init__(self, hidden_size, nhead=1, num_layers=1, max_length=5000, device=None):
        super(AuVi_Encoder, self).__init__()
        self.position_embbeding = PositionalEncoding(hidden_size, 0.1, device, max_len=max_length)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

    def forward(self, x):
        x = self.position_embbeding(x)
        output = self.transformer_encoder(x)
        return output


class RationaleSourceAligner(nn.Module):
    def __init__(self, fm_dim, rm_dim, hidden_dim, num_layers, num_heads, device):
        super(RationaleSourceAligner, self).__init__()
        self.proj_f = nn.Linear(fm_dim, hidden_dim)
        self.proj_r = nn.Linear(rm_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, 0.1, device, max_len=5)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, Fm, Rm):
        fm_proj = self.proj_f(Fm)
        rm_proj = self.proj_r(Rm)
        combined_seq = torch.cat([fm_proj.unsqueeze(1), rm_proj.unsqueeze(1)], dim=1)
        combined_seq_pos = self.pos_encoder(combined_seq)
        transformer_output = self.transformer_encoder(combined_seq_pos)
        rm_filtered = transformer_output[:, 1, :]
        return rm_filtered


class DRS_Policy(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_threshold=0.5):
        super(DRS_Policy, self).__init__()
        self.action_threshold = action_threshold
        self.actor = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(), nn.Linear(hidden_dim, 3))
        self.critic = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, state):
        logits = self.actor(state)
        values = self.critic(state)
        return logits, values

    def get_action(self, state, deterministic=False):
        logits, _ = self.forward(state)
        probs = torch.sigmoid(logits)
        dist = Bernoulli(probs)
        action = (probs > self.action_threshold).float() if deterministic else dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=1)
        return action, action_log_prob

    def evaluate(self, state, action):
        logits, state_values = self.forward(state)
        probs = torch.sigmoid(logits)
        dist = Bernoulli(probs)
        action_log_probs = dist.log_prob(action).sum(dim=1)
        dist_entropy = dist.entropy().sum(dim=1)
        return action_log_probs, state_values.squeeze(), dist_entropy


class SAFE(nn.Module):
    def __init__(self, args):
        super(SAFE, self).__init__()
        self.args = args

        self.text_encoder = BertTextEncoder(language=args.language, use_finetune=args.use_finetune)
        self.rationale_encoder = BertModel.from_pretrained(args.bert_path)

        print("Freezing parameters of rationale_encoder...")
        for param in self.rationale_encoder.parameters():
            param.requires_grad = False

        self.rationale_tokenizer = BertTokenizer.from_pretrained(args.bert_path)

        self.audio_encoder = AuVi_Encoder(hidden_size=args.audio_dim, nhead=args.a_encoder_heads,
                                          num_layers=args.a_encoder_layers, max_length=args.seq_lens[2],
                                          device=args.device)
        self.video_encoder = AuVi_Encoder(hidden_size=args.vision_dim, nhead=args.v_encoder_heads,
                                          num_layers=args.v_encoder_layers, max_length=args.seq_lens[1],
                                          device=args.device)

        self.rsa_t = RationaleSourceAligner(
            fm_dim=args.text_dim, rm_dim=args.text_dim,
            hidden_dim=args.rsa_hidden_dim, num_layers=args.rsa_num_layers,
            num_heads=args.rsa_num_heads, device=args.device
        )
        self.rsa_a = RationaleSourceAligner(
            fm_dim=args.audio_dim, rm_dim=args.text_dim,
            hidden_dim=args.rsa_hidden_dim, num_layers=args.rsa_num_layers,
            num_heads=args.rsa_num_heads, device=args.device
        )
        self.rsa_v = RationaleSourceAligner(
            fm_dim=args.vision_dim, rm_dim=args.text_dim,
            hidden_dim=args.rsa_hidden_dim, num_layers=args.rsa_num_layers,
            num_heads=args.rsa_num_heads, device=args.device
        )

        self.proj_t = nn.Linear(args.text_dim, args.hidden_dim)
        self.proj_a = nn.Linear(args.audio_dim, args.hidden_dim)
        self.proj_v = nn.Linear(args.vision_dim, args.hidden_dim)

        self.proj_r_filtered = nn.Linear(args.rsa_hidden_dim, args.hidden_dim)

        self.cross_attention_fusion = nn.MultiheadAttention(
            embed_dim=args.hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.fusion_layer_norm = nn.LayerNorm(args.hidden_dim)
        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)

        self.state_dim = args.text_dim + args.audio_dim + args.vision_dim + \
                         (args.text_dim * 3)

        self.drs_policy = DRS_Policy(self.state_dim, args.hidden_dim, args.action_threshold)

        self.fusion_dim = args.hidden_dim * 3

        self.predictor = nn.Sequential(
            nn.Linear(self.fusion_dim, args.hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(args.hidden_dim * 2),
            nn.Linear(args.hidden_dim * 2, args.num_classes)
        )

    def _encode_rationales(self, rationales_list):
        inputs = self.rationale_tokenizer(rationales_list, padding=True, truncation=True,
                                          max_length=self.args.rationale_max_len, return_tensors="pt")
        inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
        return self.rationale_encoder(**inputs).pooler_output

    def forward(self, text, audio, vision, rationale_text, rationale_vision, rationale_audio, deterministic=False):
        F_t_seq = self.text_encoder(text)
        F_t = F_t_seq[:, 0, :]

        audio_encoded_seq = self.audio_encoder(audio)
        vision_encoded_seq = self.video_encoder(vision)

        F_a = F.adaptive_avg_pool1d(audio_encoded_seq.transpose(1, 2), 1).squeeze(-1)
        F_v = F.adaptive_avg_pool1d(vision_encoded_seq.transpose(1, 2), 1).squeeze(-1)

        R_t = self._encode_rationales(rationale_text)
        R_a = self._encode_rationales(rationale_audio)
        R_v = self._encode_rationales(rationale_vision)

        state = torch.cat([
            F_t.detach(), F_a.detach(), F_v.detach(),
            R_t.detach(), R_a.detach(), R_v.detach()
        ], dim=1)

        action, action_log_prob = self.drs_policy.get_action(state, deterministic)

        R_t_filtered = self.rsa_t(F_t, R_t)
        R_a_filtered = self.rsa_a(F_a, R_a)
        R_v_filtered = self.rsa_v(F_v, R_v)

        F_t_proj = self.proj_t(F_t)
        F_a_proj = self.proj_a(F_a)
        F_v_proj = self.proj_v(F_v)
        HX = torch.stack([F_t_proj, F_a_proj, F_v_proj], dim=1)

        R_t_filt_proj = self.proj_r_filtered(R_t_filtered)
        R_a_filt_proj = self.proj_r_filtered(R_a_filtered)
        R_v_filt_proj = self.proj_r_filtered(R_v_filtered)

        action_mask = action.unsqueeze(-1)
        HR_selected = torch.stack([
            R_t_filt_proj * action_mask[:, 0, :],
            R_a_filt_proj * action_mask[:, 1, :],
            R_v_filt_proj * action_mask[:, 2, :]
        ], dim=1)

        attended_features, _ = self.cross_attention_fusion(
            query=HX,
            key=HR_selected,
            value=HR_selected
        )

        fused_features = self.fusion_layer_norm(HX + attended_features)
        final_fusion_feature = fused_features.view(fused_features.size(0), -1)
        final_fusion_feature = self.post_fusion_dropout(final_fusion_feature)

        prediction = self.predictor(final_fusion_feature)

        return prediction, state, action, action_log_prob