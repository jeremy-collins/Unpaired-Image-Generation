import torch
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import torchvision.transforms as transforms
import cv2
from model.t2ivae import T2IVAE
from tqdm import tqdm
import argparse
import numpy as np
from model.utils import *
from model.unet_utils import *
from model.config_utils import parse_config_args
import random
import wandb
import datetime
import os

def train_epoch(model, train_loader, optimizer, use_diffusion):
    model.train()
    loss_sum = 0
    avg_loss_dict = {}
    for i, (img, text, mask_img, mask_text) in enumerate(tqdm(train_loader)):
        if hasattr(config, 'WARMUP_EPOCHS') and epoch < config.WARMUP_EPOCHS:
            use_diffusion = False # using resnet decoder for vae warmup
            print('in warmup phase')

        img = img.to(device).float()
        text_input = text.to(device)          
        mask_img = mask_img.to(device)
        mask_text = mask_text.to(device)

        optimizer.zero_grad()

        output = model(img, text_input, mask_img, mask_text, use_diffusion)

        if args.debug and i % 20 == 0:
            sample_diffusion = False
            if i % 300 == 0 and use_diffusion:
                sample_diffusion = True
            print('sample diffusion: ', sample_diffusion)
            disp_img = visualize_data(img, text_input, model.tokenizer, output, config, mask_img, mask_text, model=model, sample_diffusion=sample_diffusion)
            cv2.imshow('disp_img', disp_img)
            cv2.waitKey(1)
            if sample_diffusion:
                wandb.log({"pred train image after reverse diffusion": wandb.Image(disp_img[:,:,::-1])})
            else:
                wandb.log({"pred train image after single forward pass": wandb.Image(disp_img[:,:,::-1])})
            # datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            # cv2.imwrite('logs/' + args.config + '/train_' + str(epoch) + '_' + datetime_str + '.jpg', (disp_img * 255).astype(np.uint8))

        loss_dict = criterion(output, img, text_input, mask_img, mask_text, use_diffusion)

        if torch.isnan(loss_dict['loss_total']['value']):
            print('loss is nan')
            loss_dict['loss_total']['value'] = torch.tensor(0).to(device)

        loss = loss_dict['loss_total']['value']
        print('loss_total for one batch: ', loss)
        loss.backward()

        if hasattr(config, 'CLIP_GRADS') and config.CLIP_GRADS:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.CLIP_GRADS_NORM)

        optimizer.step()
        # ema.step_ema(ema_model, model)

        loss_sum += loss.item()

        # recording the running average of the loss
        for key in loss_dict:
            if key not in avg_loss_dict: # first time
                avg_loss_dict[key] = {}
                avg_loss_dict[key]['value'] = loss_dict[key]['value']
                avg_loss_dict[key]['count'] = loss_dict[key]['count']
            else:
                avg_loss_dict[key]['value'] += loss_dict[key]['value']
                avg_loss_dict[key]['count'] += loss_dict[key]['count']

    for key in avg_loss_dict:
        avg_loss_dict[key]['value'] /= avg_loss_dict[key]['count']
        wandb.log({key + '_avg_train': avg_loss_dict[key]['value']}, step=epoch)

    return avg_loss_dict["loss_total"]["value"]

def val_epoch(model, val_loader, use_diffusion):
    model.eval()
    loss_sum = 0
    avg_loss_dict = {}
    with torch.no_grad():
        for i, (img, text, mask_img, mask_text) in enumerate(tqdm(val_loader)):
            if hasattr(config, 'WARMUP_EPOCHS') and epoch < config.WARMUP_EPOCHS:
                use_diffusion = False # using resnet decoder for vae warmup
                print('in warmup phase')

            img = img.to(device).float()
            text_input = text.to(device)          
            mask_img = mask_img.to(device)
            mask_text = mask_text.to(device)

            output = model(img, text_input, mask_img, mask_text, use_diffusion)

            if args.debug and i % 20 == 0:
                sample_diffusion = False
                if i % 200 == 0 and use_diffusion:
                    sample_diffusion = True
                disp_img = visualize_data(img, text_input, model.tokenizer, output, config, mask_img, mask_text, model=model, sample_diffusion=sample_diffusion)
                cv2.imshow('disp_img', disp_img)
                cv2.waitKey(1)
                if sample_diffusion:
                    wandb.log({"pred val image after reverse diffusion": wandb.Image(disp_img[:,:,::-1])})
                else:
                    wandb.log({"pred val image after single forward pass": wandb.Image(disp_img[:,:,::-1])})
            # datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                # datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                # cv2.imwrite(save_dir + '/val_' + str(epoch) + '_' + datetime_str + '.jpg', (disp_img * 255).astype(np.uint8))
                # print('saved val image to', save_dir + '/val_' + str(epoch) + '_' + datetime_str + '.jpg')

            loss_dict = criterion(output, img, text_input, mask_img, mask_text, use_diffusion)

            if torch.isnan(loss_dict['loss_total']['value']):
                print('loss is nan')
                loss_dict['loss_total']['value'] = torch.tensor(0).to(device)

            loss = loss_dict['loss_total']['value']

            if hasattr(config, 'CLIP_GRADS') and config.CLIP_GRADS:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.CLIP_GRADS_NORM)

            # ema.step_ema(ema_model, model)

            loss_sum += loss.item()

            # recording the running average of the loss
            for key in loss_dict:
                if key not in avg_loss_dict: # first time
                    avg_loss_dict[key] = {}
                    avg_loss_dict[key]['value'] = loss_dict[key]['value']
                    avg_loss_dict[key]['count'] = loss_dict[key]['count']
                else:
                    avg_loss_dict[key]['value'] += loss_dict[key]['value']
                    avg_loss_dict[key]['count'] += loss_dict[key]['count']

        print('avg_loss_dict: ', avg_loss_dict)
        print('avg_loss_dict: ', avg_loss_dict)
        for key in avg_loss_dict:
            avg_loss_dict[key]['value'] /= avg_loss_dict[key]['count']
            wandb.log({key + '_avg_val': avg_loss_dict[key]['value']}, step=epoch)

        print('avg_loss_dict after division: ', avg_loss_dict)

        return avg_loss_dict["loss_total"]["value"]

def get_text_loss(pred, target):
    # pred: (batch_size, seq_len, vocab_size)
    # target: (batch_size, seq_len)
    bs, seq_len, vocab_size = pred.size()
    pred = pred.view(-1, vocab_size) # (batch_size*seq_len, vocab_size)
    target = target.view(bs*seq_len) # (batch_size*seq_len)

    # cross entropy loss, ignoring padding and returning a loss vector with shape (batch_size)
    # loss = torch.nn.functional.cross_entropy(pred, target, ignore_index=0, reduction='none') # this returns (batch_size*seq_len,)
    loss = torch.nn.functional.cross_entropy(pred, target, ignore_index=0, reduction='none').view(bs, seq_len).mean(dim=1) # this returns (batch_size,)

    return loss

def get_diffusion_loss(pred_noise, noise):
    return torch.nn.functional.mse_loss(pred_noise, noise) 

def criterion(output, img, text_input, mask_img, mask_text, use_diffusion):
    # applying unit gaussian prior to combined image and text features
    combined_prior_mean = torch.zeros_like(output['combined_embedding_means']).to(device)
    combined_prior_logvar = torch.zeros_like(output['combined_embedding_logvars']).to(device)
    combined_kl_loss = kl_divergence(output['combined_embedding_means'], output['combined_embedding_logvars'], combined_prior_mean, combined_prior_logvar)
    text_loss = get_text_loss(output['pred_text'], text_input['input_ids'])
        
    if hasattr(config, 'DIFFUSION') and config.DIFFUSION:
        # getting loss vectors instead, where the vectors are the same size as the batch
        img_loss_diffusion = torch.nn.functional.mse_loss(output['pred_img_noise'], output['gt_img_noise'], reduction='none').mean(dim=(1, 2, 3))
        img_loss_L2 = torch.nn.functional.mse_loss(output['pred_img'], img, reduction='none').mean(dim=(1, 2, 3))
        img_loss_gaussian = torch.zeros(img.shape[0]).to(device)
    else:
        # applying gaussian loss between output['pred_img'] and img (t2i is not deterministic, so using max likelihood)
        img_loss_gaussian = torch.nn.GaussianNLLLoss(reduction='none')(output['pred_img_means'], img, torch.exp(output['pred_img_logvars'])).mean(dim=(1, 2, 3))
        # applying L2 loss between output['pred_img'] and pred_img (reconstruction is deterministic)
        img_loss_L2 = torch.nn.functional.mse_loss(output['pred_img'], img, reduction='none').mean(dim=(1, 2, 3))
        img_loss_diffusion = torch.zeros(img.shape[0]).to(device)

    if args.debug:
        print('img_loss_gaussian: ', img_loss_gaussian)
        print('img_loss_L2: ', img_loss_L2)
        print('text_loss: ', text_loss)
        print('kl_loss: ', combined_kl_loss)   
        print('diffusion_loss: ', img_loss_diffusion)     

        # printing the loss shapes
        print('img_loss_gaussian shape: ', img_loss_gaussian.shape)
        print('img_loss_L2 shape: ', img_loss_L2.shape)
        print('text_loss shape: ', text_loss.shape)
        print('kl_loss shape: ', combined_kl_loss.shape)
        print('diffusion_loss shape: ', img_loss_diffusion.shape)
 
    loss_dict = {
        'loss_total':
            {'value': 0, 'count': 0},
        't2i_loss_diffusion':
            {'value': 0, 'count': 0},
        't2i_loss_L2':
            {'value': 0, 'count': 0},
        't2i_loss_gaussian':
            {'value': 0, 'count': 0},
        't2t_loss':
            {'value': 0, 'count': 0},
        'kl_loss_text_only':
            {'value': 0, 'count': 0},

        'i2i_loss_diffusion':
            {'value': 0, 'count': 0},
        'i2i_loss_L2':
            {'value': 0, 'count': 0},   
        'i2i_loss_gaussian':
            {'value': 0, 'count': 0},
        'i2t_loss':
            {'value': 0, 'count': 0},
        'kl_loss_img_only':
            {'value': 0, 'count': 0},

        'no_mask_diffusion_img_loss':
            {'value': 0, 'count': 0},
        'no_mask_L2_img_loss':
            {'value': 0, 'count': 0},
        'no_mask_gaussian_img_loss':
            {'value': 0, 'count': 0},
        'no_mask_text_loss':
            {'value': 0, 'count': 0},
        'no_mask_kl_loss':
            {'value': 0, 'count': 0},
    }

    # converting all the zeros to tensors
    for key in loss_dict:
        loss_dict[key]['value'] = torch.tensor(loss_dict[key]['value']).float().to(device)
        loss_dict[key]['count'] = torch.tensor(loss_dict[key]['count']).float().to(device)

    for i in range(img.size(0)):
        if mask_img[i]:
            loss_dict['loss_total']['value'] += config.LAMBDA_TEXT * text_loss[i] + config.LAMBDA_KL * combined_kl_loss[i] # supervise text and kl
            loss_dict['loss_total']['count'] += 1

            loss_dict['t2i_loss_diffusion']['value'] += img_loss_diffusion[i]
            loss_dict['t2i_loss_diffusion']['count'] += 1

            loss_dict['t2i_loss_L2']['value'] += img_loss_L2[i]
            loss_dict['t2i_loss_L2']['count'] += 1

            loss_dict['t2i_loss_gaussian']['value'] += img_loss_gaussian[i]
            loss_dict['t2i_loss_gaussian']['count'] += 1

            loss_dict['t2t_loss']['value'] += text_loss[i]
            loss_dict['t2t_loss']['count'] += 1

            loss_dict['kl_loss_text_only']['value'] += combined_kl_loss[i]
            loss_dict['kl_loss_text_only']['count'] += 1

        elif mask_text[i]:
            if use_diffusion:
                loss_dict['loss_total']['value'] += config.LAMBDA_IMAGE * img_loss_diffusion[i] + config.LAMBDA_KL * combined_kl_loss[i] # supervise img (diffusion) and kl
                loss_dict['loss_total']['count'] += 1

            else:
                loss_dict['loss_total']['value'] += config.LAMBDA_IMAGE * img_loss_L2[i] + config.LAMBDA_KL * combined_kl_loss[i] # supervise img (diffusion) and kl
                loss_dict['loss_total']['count'] += 1

            loss_dict['i2i_loss_diffusion']['value'] += img_loss_diffusion[i]
            loss_dict['i2i_loss_diffusion']['count'] += 1

            loss_dict['i2i_loss_L2']['value'] += img_loss_L2[i]
            loss_dict['i2i_loss_L2']['count'] += 1

            loss_dict['i2i_loss_gaussian']['value'] += img_loss_gaussian[i]
            loss_dict['i2i_loss_gaussian']['count'] += 1

            loss_dict['i2t_loss']['value'] += text_loss[i]
            loss_dict['i2t_loss']['count'] += 1

            loss_dict['kl_loss_img_only']['value'] += combined_kl_loss[i]
            loss_dict['kl_loss_img_only']['count'] += 1

        else:
            if use_diffusion:
                loss_dict['loss_total']['value'] += config.LAMBDA_IMAGE * img_loss_diffusion[i] + config.LAMBDA_TEXT * text_loss[i] + config.LAMBDA_KL * combined_kl_loss[i]
                loss_dict['loss_total']['count'] += 1


            else:
                # loss_total = config.LAMBDA_IMAGE * img_loss_L2 + config.LAMBDA_TEXT * text_loss + config.LAMBDA_KL * combined_kl_loss # supervise img, text (L2), and kl
                loss_dict['loss_total']['value'] += config.LAMBDA_IMAGE * img_loss_L2[i] + config.LAMBDA_TEXT * text_loss[i] + config.LAMBDA_KL * combined_kl_loss[i]
                loss_dict['loss_total']['count'] += 1

            loss_dict['no_mask_diffusion_img_loss']['value'] += img_loss_diffusion[i]
            loss_dict['no_mask_diffusion_img_loss']['count'] += 1

            loss_dict['no_mask_L2_img_loss']['value'] += img_loss_L2[i]
            loss_dict['no_mask_L2_img_loss']['count'] += 1

            loss_dict['no_mask_gaussian_img_loss']['value'] += img_loss_gaussian[i]
            loss_dict['no_mask_gaussian_img_loss']['count'] += 1

            loss_dict['no_mask_text_loss']['value'] += text_loss[i]
            loss_dict['no_mask_text_loss']['count'] += 1

            loss_dict['no_mask_kl_loss']['value'] += combined_kl_loss[i]
            loss_dict['no_mask_kl_loss']['count'] += 1

    for key in loss_dict:
        if loss_dict[key]['count'] > 0:
            loss_dict[key]['value'] = loss_dict[key]['value'] / loss_dict[key]['count']
        else:
            loss_dict[key]['value'] = 0

    return loss_dict

def custom_collate_fn(batch):
    images, texts, mask_imgs, mask_texts = zip(*batch)

    if config.DATASET == 'coco':
        # generating a random caption index from 0 to 4 for each image
        texts = [text[random.randint(0, len(text) - 1)] for text in texts]
    elif config.DATASET == 'cifar100':
        # turning the CIFAR 100 class index into a string
        texts = [config.CIFAR100_CLASSES[text] for text in texts]
    elif config.DATASET == 'cifar10':
        # turning the CIFAR 10 class index into a string
        texts = [config.CIFAR10_CLASSES[text] for text in texts]

    text_input = model.tokenizer(texts, return_tensors="pt", padding=True, max_length=config.MAX_SEQ_LEN, truncation=True) # ["input_ids"]

    # # Define the image transformations
    # transform = transforms.Compose([
    #     transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
    # ])

    # Apply transformations to each image in the batch
    # images = [transform(image) for image in images]

    # Convert images list into a PyTorch tensor
    images = torch.stack(images)

    # Pad sequences for text
    if text_input["input_ids"].size(1) < config.MAX_SEQ_LEN:
        text_input["input_ids"] = torch.nn.functional.pad(text_input["input_ids"], (0, config.MAX_SEQ_LEN - text_input["input_ids"].shape[1]))
    else:
        text_input["input_ids"] = text_input["input_ids"][:, :config.MAX_SEQ_LEN] # truncate to max seq len

    # setting attention mask
    # ignoring padding tokens
    text_input["attention_mask"] = (text_input["input_ids"] != model.tokenizer.pad_token_id)

    # so we can access the raw text later
    # text_input["raw_text"] = torch.tensor(texts)

    return images, text_input, torch.tensor(mask_imgs), torch.tensor(mask_texts)

if __name__ == "__main__":
    config, args = parse_config_args()
    wandb.init(project='unpaired_t2i')
    wandb.config.update(config)
    wandb.config.update(args)
    
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.run.name = args.config + '_' + datetime_str

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('logs/' + args.config + datetime_str):
        save_dir = 'logs/' + args.config + datetime_str
        os.makedirs(save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T2IVAE().to(device)

    if hasattr(config, 'PRETRAINED_IMG_ENC'):
        img_enc_path = 'checkpoints/' + config.PRETRAINED_IMG_ENC + '.pt'
        model.img_encoder.load_state_dict(torch.load(img_enc_path), strict=False)
        print('loaded pretrained img encoder')

    if hasattr(config, 'CHECKPOINT') and config.CHECKPOINT is not None:
        model.load_state_dict(torch.load('checkpoints/' + config.CHECKPOINT + '.pt'))
        print('loaded checkpoint: ', config.CHECKPOINT)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    # ema = EMA(0.995)
    # ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    
    if hasattr(config, 'DIFFUSION') and config.DIFFUSION:
        use_diffusion = True
    else:
        use_diffusion = False

    if hasattr(config, 'LR_SCHEDULER') and config.LR_SCHEDULE:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.LR_SCHEDULE_STEP, gamma=config.LR_SCHEDULE_GAMMA) # multiply lr by gamma every step_size epochs
    else:
        scheduler = None

    if config.DATASET == 'coco':
        train_dataset = dset.CocoCaptions(root = 'coco/images/train2014',
                                annFile = 'coco/annotations/annotations_trainval2014/annotations/captions_train2014.json',
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((224, 224)),
                                    # transforms.Resize((56, 56)),
                                ]))
        
        val_dataset = dset.CocoCaptions(root = 'coco/images/val2014',
                                annFile = 'coco/annotations/annotations_trainval2014/annotations/captions_val2014.json',
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((224, 224)),
                                    # transforms.Resize((56, 56)),
                                ]))
    elif config.DATASET == 'cifar100':
        train_dataset = dset.CIFAR100(root='cifar100', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((32, 32))
                            ]))

        val_dataset = dset.CIFAR100(root='cifar100', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((32, 32))
                            ]))

        # train_dataset = MaskedDataset(dataset_class=dset.CIFAR100, model=model, root='cifar100', train=True, download=True,
        #                     transform=transforms.Compose([
        #                         transforms.ToTensor(),
        #                         transforms.Resize((32, 32))
        #                     ]))

        # val_dataset = MaskedDataset(dataset_class=dset.CIFAR100, model=model, root='cifar100', train=False, download=True,
        #                     transform=transforms.Compose([
        #                         transforms.ToTensor(),
        #                         transforms.Resize((32, 32))
        #                     ]))
        
        
        # loading the cifar class names from a text file
        with open('cifar100_labels.txt', 'r') as f:
            config.CIFAR100_CLASSES = f.read().splitlines()
            print('class names:', config.CIFAR100_CLASSES)

    elif config.DATASET == 'cifar10':
        # train_dataset = dset.CIFAR10(root='cifar10', train=True, download=True,
        #                     transform=transforms.Compose([
        #                         transforms.ToTensor(),
        #                         transforms.Resize((32, 32))
        #                     ]))
        
        # val_dataset = dset.CIFAR10(root='cifar10', train=False, download=True,
        #                     transform=transforms.Compose([
        #                         transforms.ToTensor(),
        #                         transforms.Resize((32, 32))
        #                     ]))
    
        train_dataset = MaskedDataset(dataset_class=dset.CIFAR10, model=model, mask_percents=config.MASK_PERCENTS_DICT, root='cifar10', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((32, 32))
                            ]))

        val_dataset = MaskedDataset(dataset_class=dset.CIFAR10, model=model, mask_percents=config.MASK_PERCENTS_DICT, root='cifar10', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((32, 32))
                            ]))
        
        # loading the cifar class names from a text file
        with open('cifar10_labels.txt', 'r') as f:
            config.CIFAR10_CLASSES = f.read().splitlines()
            print('class names:', config.CIFAR10_CLASSES)

    else:
        print('Dataset not supported')
        raise NotImplementedError

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)
    # train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS, sampler=SubsetRandomSampler(range(13))) # for debugging
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)
    # val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS, sampler=SubsetRandomSampler(range(13))) # for debugging

    best_val_loss = float('inf')

    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, use_diffusion)
        val_loss = val_epoch(model, val_loader, use_diffusion)
        wandb.log({'train_loss': train_loss, 'val_loss': val_loss}, step=epoch)
        print("Epoch: ", epoch)

        # saving model
        torch.save(model.state_dict(), 'checkpoints/' + args.config + '_latest.pt')
        print("Saved latest model")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/' + args.config + '_best.pt')
            print("Saved best model")
        if (epoch + 1) % 50 == 0 and epoch > 0:
            torch.save(model.state_dict(), 'checkpoints/' + args.config + '_epoch' + str(epoch) + '.pt')
            print("Saved model at epoch ", epoch)

    print('Number of samples: ', len(train_loader))