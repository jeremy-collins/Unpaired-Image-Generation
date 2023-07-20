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

    return torch.mean(kl_div) # summing over all elements in the batch

def get_viewable_text(token_ids, tokenizer):
    # decoding text from token ids, stopping at eos token
    viewable_text = ''

    # # getting index of eos token
    # eos_token_idx = token_ids.tolist().index(1)

    # checking if eos token is in token_ids
    if 1 in token_ids:
        eos_token_idx = token_ids.tolist().index(1)
    else:
        eos_token_idx = len(token_ids) - 1

    # decoding text
    viewable_text = tokenizer.decode(token_ids[:eos_token_idx])

    return viewable_text


def tensor_to_cv2(tensor, config=None):
    # input: tensor of shape (3, 224, 224)
    # output: numpy array of shape (224, 224, 3)
    if tensor.requires_grad:
        img = tensor.permute(1, 2, 0).cpu().detach().numpy()
    else:
        img = tensor.permute(1, 2, 0).cpu().numpy()
    # rgb to bgr
    img = img[:, :, ::-1].copy()

    if config is None or config.DATASET == 'coco':
        # resizing to 4x
        img = cv2.resize(img, (0, 0), fx=4, fy=4)
    elif config.DATASET == 'cifar100':
        # resizing to 28x
        img = cv2.resize(img, (0, 0), fx=28, fy=28)

    return img

def visualize_data(img_input, text_input, tokenizer, output=None, config=None, mask_img=False, mask_text=False):
    # visualize the data given the inputs or outputs
    # img: [batch_size, 3, 224, 224]
    # text_input: [batch_size, max_seq_len]
    # output: [batch_size, 3, 224, 224]
    # output: [batch_size, max_seq_len, vocab_size]

    # visualizing image and caption
    if mask_img:
        gt_img = tensor_to_cv2(torch.zeros_like(img_input[0]), config)
    else:
        gt_img = tensor_to_cv2(img_input[0], config)

    # decoding text from token ids
    # viewable_text = model.tokenizer.decode(text_input["input_ids"][0], skip_special_tokens=True)
    if mask_text:
        viewable_text_gt = ''
    else:
        viewable_text_gt = get_viewable_text(text_input["input_ids"][0], tokenizer)

    gt_img = cv2.putText(gt_img, 'ground truth: ' + viewable_text_gt, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # cv2.imshow('input img', gt_img)

    if output is not None:
        # visualizing predicted image and caption
        img_from_img = tensor_to_cv2(output['pred_img'][0], config)

        # getting predicted token ids
        pred_token_ids = torch.argmax(output['pred_text'][0], dim=1)
        viewable_text_pred = get_viewable_text(pred_token_ids, tokenizer)

        img_from_img = cv2.putText(img_from_img, 'VAE: ' + viewable_text_pred, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # cv2.imshow('pred_img_from_img', img_from_img)

        img_from_text = tensor_to_cv2(output['pred_img_t2i'][0], config)
        img_from_text = cv2.putText(img_from_text, 't2i prompt: ' + viewable_text_gt, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        print('i2t shape: ', output['pred_text_i2t'].shape)
        pred_i2t_token_ids = torch.argmax(output['pred_text_i2t'][0], dim=1)
        text_from_img = get_viewable_text(pred_i2t_token_ids, tokenizer)
        
        gt_img = cv2.putText(gt_img, 'pred caption: ' + text_from_img, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # cv2.imshow('pred_img_t2i', img_from_text)
        disp_img = np.concatenate((gt_img, img_from_img, img_from_text), axis=1)
    else:
        disp_img = gt_img

    return disp_img

class MaskedDataset(Dataset):
    def __init__(self, dataset_class, model, *args, **kwargs):
        self.dataset = dataset_class(*args, **kwargs)
        self.model = model

    def __getitem__(self, index):
        # Make a deep copy of the image and caption
        image, caption = copy.deepcopy(self.dataset[index])

        # Apply the masking
        mask_img = random.randint(0, 1)
        mask_text = random.randint(0, 1)
        if mask_img:
            image = torch.zeros_like(image)
        if mask_text:
            print('caption: ', caption)
            # caption["input_ids"] = torch.full_like(caption["input_ids"], self.model.tokenizer.eos_token_id)
            caption = self.model.tokenizer.eos_token_id

        
        return image, caption

    def __len__(self):
        return len(self.dataset)
    
def get_masks():
    # 3 possible states
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