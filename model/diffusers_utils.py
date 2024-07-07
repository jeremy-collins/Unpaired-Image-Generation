import datetime
import math
import os

import cv2
import matplotlib.pyplot as plt
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm import tqdm

# sourced from
# https://github.com/dome272/Diffusion-Models-pytorch/blob/main/ddpm_conditional.py
class Diffusion:
    # def __init__(self, noise_steps=100, beta_start=1e-4, beta_end=0.02, scheduler='squaredcos_cap_v2'):
    def __init__(self, noise_steps=100, beta_start=1e-4, beta_end=0.02, img_size=32, scheduler='squaredcos_cap_v2'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.diffusers_scheduler = DDPMScheduler(
            num_train_timesteps=noise_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=scheduler,
            clip_sample=True,
            variance_type='fixed_small',
            prediction_type='epsilon',
            )

        if scheduler == 'squaredcos_cap_v2':
            self.beta = self.squaredcos_cap_v2(self.noise_steps, max_beta=self.beta_end)
        elif scheduler == 'cosine':
            self.beta = self.cosine_beta_schedule(self.noise_steps)
        elif scheduler == 'linear':
            self.beta = self.linear_schedule()
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler}")

        self.alpha = (1. - self.beta).to(self.device)
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(self.device)
        self.beta = self.beta.to(self.device)

    def linear_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    # Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
    def squaredcos_cap_v2(
        self,
        num_diffusion_timesteps,
        max_beta=0.999,
        alpha_transform_type="cosine",
    ):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
        (1-beta) over time from t = [0,1].

        Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
        to that part of the diffusion process.


        Args:
            num_diffusion_timesteps (`int`): the number of betas to produce.
            max_beta (`float`): the maximum beta to use; use values lower than 1 to
                        prevent singularities.
            alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                        Choose from `cosine` or `exp`

        Returns:
            betas (`torch.Tensor`): the beta schedule to use for the diffusion process.
        """
        if alpha_transform_type == "cosine":

            def alpha_bar_fn(t):
                return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

        elif alpha_transform_type == "exp":

            def alpha_bar_fn(t):
                return math.exp(t * -12.0)

        else:
            raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
        return torch.tensor(betas, dtype=torch.float32)

    def noise_images(self, x, t, clamp=False, skip_first=False):
        noise = torch.randn(x.shape, device=x.device)
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        print('x:', x.shape)
        print('noise:', noise.shape)
        print('t:', t.shape)

        noised_x = self.diffusers_scheduler.add_noise(
            x, noise, t)

        return noised_x, noise

    def sample_timesteps(self, n):
        # return torch.randint(low=1, high=self.noise_steps, size=(n,)).long()
        return torch.randint(
            0, self.diffusers_scheduler.config.num_train_timesteps, 
            (n,)
        ).long()
    
    # def sample(self, model, queries, labels, size, n=1, cfg_scale=0, vis_img=None, name=''):
    def sample(self, model, n, labels, cfg_scale=3):
        # print("sample size:", size) # (seq_len, num_tracks, 2)
        print(f"Sampling new image....")
        # if not os.path.exists('results/diffusion_vis'):
            # os.makedirs('results/diffusion_vis')
            
        # making a video using cv2
        # datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # video = cv2.VideoWriter(f'results/diffusion_vis/{name}_{datetime_str}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (128, 128)) # TODO: use config for image size

        with torch.no_grad():
            # x = torch.randn((n, size[0], size[1])).to(self.device) # (n, seq_len, num_tracks, 2) in range (-1, 1)
            # x = torch.randn((n, size[0], size[1], size[2])).to(self.device) # (n, seq_len, num_tracks, 2) in range (-1, 1)
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            print('x:', x.shape)
            print('labels:', labels.shape)
            print("sampling from diffusion model...")
            for t in tqdm(self.diffusers_scheduler.timesteps):
                t = t.to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                x = self.diffusers_scheduler.step(
                predicted_noise, t, x.clone()
                ).prev_sample

                # display the image
                img = (x.clamp(-1, 1) + 1) / 2
                img = (img * 255).type(torch.uint8)
                img = img[0].permute(1, 2, 0).cpu().numpy()
                # rgb to bgr
                img = img[:, :, ::-1].copy()
                img = cv2.resize(img, (0, 0), fx=28, fy=28, interpolation=cv2.INTER_NEAREST)
                cv2.imshow('disp_img', img)
                cv2.waitKey(1)
        

        # model.train()
        return x