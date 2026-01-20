if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from termcolor import cprint
from pcdp.workspace.base_workspace import BaseWorkspace
# from pcdp.policy.diffusion_unet_pointcloud_policy import DiffusionUnetPointCloudPolicy
from pcdp.dataset.base_dataset import BasePointCloudDataset
# from pcdp.env_runner.base_pointcloud_runner import BasePointCloudRunner
from pcdp.common.checkpoint_util import TopKCheckpointManager
from pcdp.common.json_logger import JsonLogger
from pcdp.common.pytorch_util import dict_apply, optimizer_to
from pcdp.model.diffusion.ema_model import EMAModel
from pcdp.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetPointCloudWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir = None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        """
        self.model: DiffusionUnetPointCloudPolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: DiffusionUnetPointCloudPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        
        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params= self.model.parameters())
        """

        # configure training state
        self.global_step = 0
        self.epoch = 0
    
    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                cprint(f"Resuming from checkpoint {lastest_ckpt_path}", "yellow", attrs=["bold"])
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BasePointCloudDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BasePointCloudDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()
        # configure validataion dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        
        """
        # self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)
        
        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader)*cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch = self.global_step-1
        )
        
        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model = self.ema_model)
        
        # configure env
        env_runner: BasePointCloudRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir = self.output_dir)
        assert isinstance(env_runner, BasePointCloudRunner)

        
        # configure logging
        wandb_run = wandb.init(
            dir = str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir
            }
        )
        
        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir = os.path.join(self.output_dir, 'checkpoints')
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        # save batch or sampling
        train_sampling_batch = None
        """
        device = torch.device(cfg.training.device)

        import time

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
        
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ====== train for this epoch ======
                """
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_gard_(False)
                """
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        # print(f"batch_obs: {batch['obs'].keys()}")
                        # print(f"debug: {batch['obs']['align_timestamp'].shape}")
                        print(f"batch idx: {batch_idx}")
                        for idx in range(batch['action'].shape[0]):
                            print(f"{idx} time: {batch['obs']['align_timestamp'][idx]}")
                            print(f"{idx} action: {batch['action'][idx][:3]}")
                        time.sleep(10)

                            


                        
        self.global_step += 1
        self.epoch += 1
        


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetPointCloudWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
