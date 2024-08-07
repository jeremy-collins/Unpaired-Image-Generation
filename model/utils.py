import torch
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
from model.t2ivae import T2IVAE
from tqdm import tqdm
import argparse
import numpy as np
from model.utils import *
import random
import copy
import wandb

def kl_divergence(mu1, logvar1, mu2, logvar2):
    # kl divergence from means and logvars
    # # kl_div = (q_log_var-p_log_var + (jnp.exp(p_log_var)+(p_mean-q_mean)**2)/jnp.exp(q_log_var)-1)/2 

    # sanity check
    # var1 = torch.exp(logvar1)
    # var2 = torch.exp(logvar2)
    # kl_div = 0.5 * (torch.log(var2 / var1) + (var1 + (mu1 - mu2)**2) / (var2) - 1)
    # print('kl test: ', kl_div.sum())

    # Calculate KL divergence
    kl_div = 0.5 * (logvar2 - logvar1 + (torch.exp(logvar1) + (mu1 - mu2)**2) / torch.exp(logvar2) - 1) # (batch_size, hidden_dim)

    # return torch.mean(kl_div) # summing over all elements in the batch
    return kl_div.mean(dim=1) # shape: (batch_size,)

def get_viewable_text(token_ids, tokenizer):
    # decoding text from token ids, stopping at eos token
    viewable_text = ''

    # checking if eos token is in token_ids
    if 1 in token_ids:
        eos_token_idx = token_ids.tolist().index(1)
    else:
        eos_token_idx = len(token_ids) - 1

    # decoding text
    viewable_text = tokenizer.decode(token_ids[:eos_token_idx])

    return viewable_text


def tensor_to_cv2(tensor, config=None):
    # input: tensor of shape (3, img_size, img_size)
    # output: numpy array of shape (img_size, img_size, 3)

    if tensor.requires_grad:
        img = tensor.permute(1, 2, 0).cpu().detach().numpy()
    else:
        img = tensor.permute(1, 2, 0).cpu().numpy()

    # rgb to bgr
    img = img[:, :, ::-1].copy()

    if config is None or config.DATASET == 'coco':
        # resizing to 4x
        img = cv2.resize(img, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    elif config.DATASET == 'cifar100':
        # resizing to 28x
        img = cv2.resize(img, (0, 0), fx=28, fy=28, interpolation=cv2.INTER_NEAREST)
    elif config.DATASET == 'cifar10':
        # resizing to 28x
        img = cv2.resize(img, (0, 0), fx=28, fy=28, interpolation=cv2.INTER_NEAREST)
    else:
        raise ValueError('invalid dataset')
    
    return img

def visualize_data(img_input, text_input, tokenizer, output=None, config=None, mask_img=False, mask_text=False, model=None, sample_diffusion=False):
    gt_img = tensor_to_cv2(img_input[0], config)
    zero_img = tensor_to_cv2(torch.zeros_like(img_input[0]), config)
    text_gt = get_viewable_text(text_input["input_ids"][0], tokenizer)
    
    if output is not None:
        if hasattr(config, 'DIFFUSION') and config.DIFFUSION:
            noised_img = img_input[0] + output['gt_img_noise'][0]
            # gt_img = tensor_to_cv2(noised_img, config) # showing ground truth image with noise
            if sample_diffusion:
                img_output = model.diffusion.sample(model.unet, 1, model.combined_embedding[0], cfg_scale=0)[0]
                # img_output = model.diffusion.sample(model.unet, 1, model.combined_embedding[0], cfg_scale=3)
                # wandb.log({"predicted image after reverse diffusion": wandb.Image(img_output)})
            else:
                img_output = output['pred_img'][0]
                # wandb.log({"predicted image after single forward pass": wandb.Image(img_output)})

            gt_img = cv2.putText(gt_img, 'noise step: ' + str(output['noise_step'][0].item()), (0, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 1), 2)
            zero_img = cv2.putText(zero_img, 'noise step: ' + str(output['noise_step'][0].item()), (0, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 1), 2)

            img_output = tensor_to_cv2(img_output, config)
            # img_output = tensor_to_cv2(output['pred_img'][0], config)
            if output['unconditioned']:
                gt_img = cv2.putText(gt_img, '[unconditioned]', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 1), 2)
                img_output = cv2.putText(img_output, '[unconditioned]', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 1), 2)
                zero_img = cv2.putText(zero_img, '[unconditioned]', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 1), 2)

        else:
            # visualizing predicted image and caption
            # img_output = tensor_to_cv2(output['pred_img'][0], config)
            img_output = tensor_to_cv2(output['pred_img_means'][0], config)

        pred_token_ids = torch.argmax(output['pred_text'][0], dim=1)
        text_output = get_viewable_text(pred_token_ids, tokenizer)


        if not mask_img[0] and not mask_text[0]: # vae
            disp_gt_img = cv2.putText(gt_img, 'ground truth: ' + text_gt, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 1), 2)
            disp_pred_img = cv2.putText(img_output, 'vae output: ' + text_output, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 1), 2)
        
        elif mask_img[0] and not mask_text[0]: # t2i
            disp_gt_img = cv2.putText(zero_img, 'ground truth: ' + text_gt, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 1), 2)
            disp_pred_img = cv2.putText(img_output, 'pred image from text (' + text_gt + ')', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 1), 2)

        elif mask_text[0] and not mask_img[0]: # i2t
            disp_gt_img = cv2.putText(gt_img, 'ground truth: ' + text_gt, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 1), 2)
            disp_pred_img = cv2.putText(zero_img, 'pred text from image: ' + text_output, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 1), 2)

        else:
            raise ValueError('invalid mask_img and mask_text combination')

        # disp_img = np.concatenate((gt_img, img_from_img, img_from_text), axis=1)
        disp_img = np.concatenate((disp_gt_img, disp_pred_img), axis=1)

    else:
        # visualizing only ground truth image and caption
        disp_gt_img = cv2.putText(gt_img, 'ground truth: ' + text_gt, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        disp_img = disp_gt_img

    return disp_img

class MaskedDataset(Dataset):
    def __init__(self, dataset_class, model, mask_percents, *args, **kwargs):
        self.dataset = dataset_class(*args, **kwargs)
        self.model = model
        self.mask_percents = mask_percents

    def __getitem__(self, index):
        # Make a deep copy of the image and caption
        image, caption = copy.deepcopy(self.dataset[index])

        # we want the masks to be deterministic, so we can set the seed to the index
        random.seed(index)
        randnum = random.random() # given the index, this will always return the same random number in the range (0, 1)
        mask_img, mask_text = False, False

        if randnum < self.mask_percents['mask_img']:
            mask_img = True
        elif randnum >= self.mask_percents['mask_img'] and randnum < (self.mask_percents['mask_img'] + self.mask_percents['mask_text']):
            # caption["input_ids"] = torch.full_like(caption["input_ids"], self.model.tokenizer.eos_token_id)
            # caption["input_ids"] = torch.zeros_like(caption["input_ids"])
            mask_text = True
        elif randnum >= (self.mask_percents['mask_img'] + self.mask_percents['mask_text']):
            pass
        
        return image, caption, int(mask_img), int(mask_text)

    def __len__(self):
        return len(self.dataset)
    
def get_masks(warmup=False): # TODO: add warmup condition, move out of trainer
    # 3 possible states
    # if warmup:
    #     mask_state = random.randint(0, 2)
    # else:
    #     mask_state = random.randint(0, 1) # only mask one thing at a time if not warmup

    mask_state = random.randint(0, 2)

    if mask_state == 0:
        # mask image
        mask_img = True
        mask_text = False
    elif mask_state == 1:
        # mask text
        mask_img = False
        mask_text = True
    else:
        # mask neither
        mask_img = False
        mask_text = False

    return mask_img, mask_text