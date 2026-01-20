# diffusion_PCM_policy.py
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from termcolor import cprint
from pcdp.policy.base_pointcloud_policy import BasePointCloudPolicy
from pcdp.model.diffusion.RISE_diffusionUNet import DiffusionUNetPolicy
from pcdp.model.vision.Sparse3DEncoder import Sparse3DEncoder
from pcdp.policy.RISE_transformer import Transformer
import MinkowskiEngine as ME
from pcdp.model.common.normalizer import LinearNormalizer


class PCMPolicy(BasePointCloudPolicy):
    def __init__(
            self,
            num_action = 20,
            input_dim = 7,
            obs_feature_dim = 512,
            action_dim = 10,
            hidden_dim = 512,
            nheads = 8,
            num_encoder_layers = 4,
            num_decoder_layers = 1,
            dim_feedforward = 2048,
            dropout = 0.1,
            # ---- 입력 전처리 ----
            enable_c_gate: bool = False,         # True면 rgb *= c
            # ---- 인코더 토큰 상한(YAML에서 조절) ----
            encoder_max_num_token: int = 100,
            ):
        super().__init__()
        num_obs = 1
        robot_obs_dim = 10
        print(f"input_dim: {input_dim}")

        # single encoder
        self.encoder = Sparse3DEncoder(input_dim=input_dim, output_dim=obs_feature_dim)
        # rise
        self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, obs_feature_dim + robot_obs_dim)
        # self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, obs_feature_dim)
        self.readout_embed = nn.Embedding(1, hidden_dim)
        self.normalizer = LinearNormalizer()
        # policy
        self.enable_c_gate=enable_c_gate
        self.encoder_max_num_token = encoder_max_num_token
        

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    @torch.no_grad()
    def _apply_normalize_and_gate(self, cloud: ME.SparseTensor) -> ME.SparseTensor:
        """RGB 정규화(+선택적 c-게이팅)만 수행."""
        feats, coords = cloud.F, cloud.C
        colors = feats[:, 3:6]
        c = feats[:, 6:7]

        # RGB normalize
        colors = self.normalizer['pointcloud_color'].normalize(colors)

        # c-gate (신선할수록 영향↑)
        if self.enable_c_gate:
            colors = colors * c

        new_feats = torch.cat([feats[:, :3], colors, c], dim=1)
        return ME.SparseTensor(features=new_feats, coordinates=coords, device=cloud.device)


    def forward(self, cloud: ME.SparseTensor, actions=None, robot_obs=None, batch_size=24):
        assert robot_obs is not None, "PCMPolicy expects robot_obs (B,10) or (B,T,10)."
        if robot_obs.dim() == 2:  # [B,10] -> [B,1,10]
            robot_obs = robot_obs.unsqueeze(1)

        dev = cloud.F.device if hasattr(cloud, "F") else cloud.device
        if self.normalizer is not None:
            self.normalizer.to(dev)

        # Normalize inputs if normalizer is set
        if self.normalizer is not None:
            # Normalize point cloud colors
            cloud = self._apply_normalize_and_gate(cloud)

            # Normalize low-dim robot_obs
            # robot_obs: (B, T, 10) where T=n_obs_steps
            obs_trans = robot_obs[:, :, :3]
            obs_rot = robot_obs[:, :, 3:9]
            obs_grip = robot_obs[:, :, 9:10]
            norm_obs_trans = self.normalizer['obs_translation'].normalize(obs_trans)
            norm_obs_grip = self.normalizer['obs_gripper'].normalize(obs_grip)
            robot_obs = torch.cat([norm_obs_trans, obs_rot, norm_obs_grip], dim=-1)

            if actions is not None:
                # Normalize actions for training
                # actions: (B, H, 10) where H=horizon
                action_trans = actions[:, :, :3]
                action_rot = actions[:, :, 3:9]
                action_grip = actions[:, :, 9:10]
                norm_action_trans = self.normalizer['action_translation'].normalize(action_trans)
                norm_action_grip = self.normalizer['action_gripper'].normalize(action_grip)
                actions = torch.cat([norm_action_trans, action_rot, norm_action_grip], dim=-1)
        

        # 4) 단일 인코더 (최종 토큰 상한은 encoder_max_num_token)
        src, pos, pad = self.encoder(
            cloud, batch_size=batch_size, max_num_token=self.encoder_max_num_token
        )  # shapes: [B, N, D], [B, N, D], [B, N]

        # 5) RISE Transformer + readout
        readout_seq = self.transformer(src, pad, self.readout_embed.weight, pos)[-1]  # [B,1,D]
        readout = readout_seq[:, 0, :]                                                # [B,D]

        
        global_cond = torch.cat([readout, robot_obs[:,0,:]], dim =-1)

        if actions is not None:
            # training mode
            loss = self.action_decoder.compute_loss(global_cond, actions)
            return loss
        else:
            # inference mode
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(global_cond)

            # un-normalize action prediction
            if self.normalizer is not None:
                action_trans = action_pred[:, :, :3]
                action_rot = action_pred[:, :, 3:9]
                action_grip = action_pred[:, :, 9:10]

                unnorm_action_trans = self.normalizer['action_translation'].unnormalize(action_trans)
                unnorm_action_grip = self.normalizer['action_gripper'].unnormalize(action_grip)

                action_pred = torch.cat([unnorm_action_trans, action_rot, unnorm_action_grip], dim=-1)

            return action_pred