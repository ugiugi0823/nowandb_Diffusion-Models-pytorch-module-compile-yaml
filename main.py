import os
import gc
import copy
import time 
import argparse
import yaml
import torch
import wandb

import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from GPUtil import showUtilization as gpu_usage

from utils import plot_images, save_images, setup_logging, get_data, save_images2
from modules import UNet_conditional, EMA
import logging



logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train_model(cfg):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # get_data
    image_size = cfg['DATA']['image_size']
    dataset_path = cfg['DATA']['dataset_path']
    batch_size = cfg['TRAIN']['batch_size']
    dataloader = get_data(image_size, dataset_path, batch_size)

    # setup_logging
    # run_name = cfg['DATA']['run_name']
    setup_logging(cfg)

    # device
    device = cfg['TRAIN']['device']
    
    # show image in wandb
    columns=["id", "image", "image_ema"]
    
    model = UNet_conditional(num_classes=cfg['DATA']['num_classes'])
    model = torch.nn.DataParallel(model).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg['TRAIN']['lr'])
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=cfg['DATA']['image_size'], device=device)
    logger = SummaryWriter(os.path.join("runs", cfg['DATA']['run_name']))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(cfg['TRAIN']['epochs']):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            print('🐬'*10,t.shape)
            x_t, noise = diffusion.noise_images(images, t)
            print('🐬'*10,x_t.shape)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())

            # wandb.log({"MSE": loss.item()})
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:
            gpu_usage()
            labels = torch.arange(cfg['DATA']['num_classes']).long().to(device)
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)

            # 시각화
            plot_images(sampled_images)
            save_images(sampled_images, os.path.join("./results", cfg['DATA']['run_name'], f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("./results", cfg['DATA']['run_name'], f"{epoch}_ema.jpg"))
            
            # 이미지 그리드 저장
            img = save_images2(sampled_images)
            img_ema = save_images2(ema_sampled_images)

            # test_table = wandb.Table(columns=columns)
            # test_table.add_data(epoch, wandb.Image(img), wandb.Image(img_ema))
            # wandb.log({"test_predictions" : test_table})

            # 모델 저장
            torch.save(model.state_dict(), os.path.join("./models", cfg['DATA']['run_name'], f"ckpt_{epoch}_.pt"))
            torch.save(ema_model.state_dict(), os.path.join("./models", cfg['DATA']['run_name'], f"ema_ckpt_{epoch}_.pt"))
            torch.save(optimizer.state_dict(), os.path.join("./models", cfg['DATA']['run_name'], f"optim_{epoch}_.pt"))


def init():
    # configs 
    parser = argparse.ArgumentParser(description='iyaho')
    parser.add_argument('--yaml_config', type=str, default='./configs/wta.yaml', help='exp config file')    
    args = parser.parse_args()
    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)

    # cfg['SAVE']['savedir'] = savedir
    # with open(f"{savedir}/config.yaml",'w') as f:
    #   yaml.dump(cfg,f)

    # wandb.login(key='')
    
    # run = wandb.init(
    #   # Set the project where this run will be logged
    #   project=cfg['DATA']['day'],
    #   # Track hyperparameters and run metadata
    #   config=cfg,
    # )
    
    return cfg 
    
    # # save point 
    # train = True
    # try:
    #     train = input('시작할까요? Enter Yes')
    #     print('시자아악!')
    # except:
    #     pass
    

if __name__ == '__main__':
    gc.collect()
    end = time.time() 
    cfg = init()
    train_model(cfg)
    print(time.time() -end)
