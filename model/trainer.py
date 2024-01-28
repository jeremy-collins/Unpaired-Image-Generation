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

# import # wandb
import datetime
import os


class Trainer:
    def __init__(self) -> None:
        self.config, self.args = parse_config_args()

        # wandb.init(project='unpaired_t2i')
        # wandb.config.update(config)
        # wandb.config.update(args)

        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # wandb.run.name = args.config + '_' + datetime_str

        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")
        if not os.path.exists("logs"):
            os.makedirs("logs")
        if not os.path.exists("logs/" + self.args.config + datetime_str):
            self.save_dir = "logs/" + self.args.config + datetime_str
            os.makedirs(self.save_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T2IVAE().to(self.device)

        if hasattr(self.config, "PRETRAINED_IMG_ENC"):
            img_enc_path = "checkpoints/" + self.config.PRETRAINED_IMG_ENC + ".pt"
            self.model.img_encoder.load_state_dict(
                torch.load(img_enc_path), strict=False
            )
            print("loaded pretrained img encoder")

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.LEARNING_RATE
        )

        if hasattr(self.config, "DIFFUSION") and self.config.DIFFUSION:
            self.use_diffusion = True
        else:
            self.use_diffusion = False

        if hasattr(self.config, "LR_SCHEDULER") and self.config.LR_SCHEDULE:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.LR_SCHEDULE_STEP,
                gamma=self.config.LR_SCHEDULE_GAMMA,
            )  # multiply lr by gamma every step_size epochs
        else:
            self.scheduler = None

        self.train_dataset, self.val_dataset = self.load_datasets()

    def load_datasets(self):
        if self.config.DATASET == "coco":
            train_dataset = dset.CocoCaptions(
                root="coco/images/train2014",
                annFile="coco/annotations/annotations_trainval2014/annotations/captions_train2014.json",
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Resize((224, 224)),
                        # transforms.Resize((56, 56)),
                    ]
                ),
            )

            val_dataset = dset.CocoCaptions(
                root="coco/images/val2014",
                annFile="coco/annotations/annotations_trainval2014/annotations/captions_val2014.json",
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Resize((224, 224)),
                        # transforms.Resize((56, 56)),
                    ]
                ),
            )
        elif self.config.DATASET == "cifar100":
            train_dataset = dset.CIFAR100(
                root="cifar100",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Resize((32, 32))]
                ),
            )

            self.val_dataset = dset.CIFAR100(
                root="cifar100",
                train=False,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Resize((32, 32))]
                ),
            )
            # loading the cifar class names from a text file
            with open("cifar100_labels.txt", "r") as f:
                self.config.CIFAR100_CLASSES = f.read().splitlines()
                print("class names:", self.config.CIFAR100_CLASSES)
        elif self.config.DATASET == "cifar10":
            train_dataset = dset.CIFAR10(
                root="cifar10",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Resize((32, 32))]
                ),
            )

            val_dataset = dset.CIFAR10(
                root="cifar10",
                train=False,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Resize((32, 32))]
                ),
            )

            # loading the cifar class names from a text file
            with open("cifar10_labels.txt", "r") as f:
                self.config.CIFAR10_CLASSES = f.read().splitlines()
                print("class names:", self.config.CIFAR10_CLASSES)

        else:
            print("Dataset not supported")
            raise NotImplementedError

        return train_dataset, val_dataset

    def custom_collate_fn(self, batch):
        images, texts = zip(*batch)

        if self.config.DATASET == "coco":
            # generating a random caption index from 0 to 4 for each image
            texts = [text[random.randint(0, len(text) - 1)] for text in texts]
        elif self.config.DATASET == "cifar100":
            # turning the CIFAR 100 class index into a string
            texts = [self.config.CIFAR100_CLASSES[text] for text in texts]
        elif self.config.DATASET == "cifar10":
            # turning the CIFAR 10 class index into a string
            texts = [self.config.CIFAR10_CLASSES[text] for text in texts]

        text_input = self.model.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            max_length=self.config.MAX_SEQ_LEN,
            truncation=True,
        )  # ["input_ids"]

        # Convert images list into a PyTorch tensor
        images = torch.stack(images)

        # Pad sequences for text
        if text_input["input_ids"].size(1) < self.config.MAX_SEQ_LEN:
            text_input["input_ids"] = torch.nn.functional.pad(
                text_input["input_ids"],
                (0, self.config.MAX_SEQ_LEN - text_input["input_ids"].shape[1]),
            )
        else:
            text_input["input_ids"] = text_input["input_ids"][
                :, : self.config.MAX_SEQ_LEN
            ]  # truncate to max seq len

        # setting attention mask
        # ignoring padding tokens
        text_input["attention_mask"] = (
            text_input["input_ids"] != self.model.tokenizer.pad_token_id
        )

        return images, text_input

    def train(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            collate_fn=self.custom_collate_fn,
            num_workers=self.config.NUM_WORKERS,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            collate_fn=self.custom_collate_fn,
            num_workers=self.config.NUM_WORKERS,
        )

        best_val_loss = float("inf")
        for epoch in range(self.config.NUM_EPOCHS):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.val_epoch(val_loader, epoch)

            # wandb.log({'train_loss': train_loss, 'val_loss': val_loss}, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    self.model.state_dict(),
                    "checkpoints/" + self.args.config + "_best.pt",
                )
                print("Saved best model")
            if epoch % 10 == 0 and epoch > 0:
                torch.save(
                    self.model.state_dict(),
                    "checkpoints/" + self.args.config + "_epoch" + str(epoch) + ".pt",
                )
                print("Saved model at epoch ", epoch)

            if self.scheduler is not None:
                self.scheduler.step()

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        loss_sum = 0
        avg_loss_dict = {}
        for i, (img, text) in enumerate(tqdm(train_loader)):
            print("using diffusion: ", self.use_diffusion)
            if (
                hasattr(self.config, "WARMUP_EPOCHS")
                and epoch < self.config.WARMUP_EPOCHS
            ):
                self.use_diffusion = False  # using resnet decoder for vae warmup
                print("in warmup phase")
                mask_img, mask_text = False, False
            elif self.config.MASKING:
                if hasattr(self.config, "DIFFUSION") and self.config.DIFFUSION:
                    self.use_diffusion = True
                print("in masking phase")
                mask_img, mask_text = get_masks(warmup=False)
            else:
                mask_img, mask_text = False, False

            # print('text: ', text)
            img = img.to(self.device).float()
            # text_input = model.tokenizer(text, return_tensors="pt", padding=True).to(device)
            text_input = text.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(
                img, text_input, mask_img, mask_text, self.use_diffusion
            )

            if self.args.debug and i % 20 == 0:
                sample_diffusion = False
                if i % 100 == 0 and self.use_diffusion:
                    sample_diffusion = True
                print("sample diffusion: ", sample_diffusion)
                disp_img = visualize_data(
                    img,
                    text_input,
                    self.model.tokenizer,
                    output,
                    self.config,
                    mask_img,
                    mask_text,
                    model=self.model,
                    sample_diffusion=sample_diffusion,
                )
                cv2.imshow("disp_img", disp_img)
                cv2.waitKey(1)
                datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                # cv2.imwrite('logs/' + args.config + '/train_' + str(epoch) + '_' + datetime_str + '.jpg', (disp_img * 255).astype(np.uint8))

            loss_dict = self.criterion(
                output, img, text_input, mask_img, mask_text, self.use_diffusion
            )
            # quit if loss is nan
            if torch.isnan(loss_dict["loss_total"]):
                print("loss is nan")
                # quit()
                loss_dict["loss_total"] = torch.tensor(0).to(self.device)

            loss = loss_dict["loss_total"]
            loss.backward()

            if hasattr(self.config, "CLIP_GRADS") and self.config.CLIP_GRADS:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.config.CLIP_GRADS_NORM
                )

            self.optimizer.step()
            # ema.step_ema(ema_model, model)

            loss_sum += loss.item()
            # for key in loss_dict:
            #     # wandb.log({key + '_train': loss_dict[key].item()})

            # recording the running average of the loss
            for key in loss_dict:
                if key not in avg_loss_dict:  # first time
                    avg_loss_dict[key] = loss_dict[key].item()
                else:
                    avg_loss_dict[key] += loss_dict[key].item()

        for key in avg_loss_dict:
            if key == "loss_total":
                avg_loss_dict[key] /= len(train_loader)
            elif self.config.MASKING:
                avg_loss_dict[key] /= (
                    len(train_loader) / 3
                )  # each loss only occurs 1/3 of the time. TODO: make this less hacky
            else:
                avg_loss_dict[key] /= len(train_loader)

            # wandb.log({key + '_avg_train': avg_loss_dict[key]}, step=epoch)

        return loss_sum / len(train_loader)

    def val_epoch(self, val_loader, epoch):
        self.model.eval()
        loss_sum = 0
        avg_loss_dict = {}
        with torch.no_grad():
            for i, (img, text) in enumerate(tqdm(val_loader)):
                if self.config.MASKING:
                    mask_img, mask_text = get_masks()
                else:
                    mask_img, mask_text = False, False

                use_diffusion = False  # using resnet decoder for vae warmup
                if (
                    hasattr(self.config, "WARMUP_EPOCHS")
                    and epoch < self.config.WARMUP_EPOCHS
                ):
                    use_diffusion = False  # using resnet decoder for vae warmup
                elif hasattr(self.config, "DIFFUSION") and self.config.DIFFUSION:
                    use_diffusion = True

                img = img.to(self.device).float()
                text_input = text.to(self.device)

                output = self.model(img, text_input, mask_img, mask_text, use_diffusion)

                if self.args.debug and i % 20 == 0:
                    sample_diffusion = False
                    if i % 100 == 0 and use_diffusion:
                        sample_diffusion = True
                    disp_img = visualize_data(
                        img,
                        text_input,
                        self.model.tokenizer,
                        output,
                        self.config,
                        mask_img,
                        mask_text,
                        model=self.model,
                        sample_diffusion=sample_diffusion,
                    )
                    cv2.imshow("disp_img", disp_img)
                    cv2.waitKey(1)
                    datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    # cv2.imwrite('logs/val_' + str(epoch) + '_' + datetime_str + '.jpg', (disp_img * 255).astype(np.uint8))
                    cv2.imwrite(
                        self.save_dir
                        + "/val_"
                        + str(epoch)
                        + "_"
                        + datetime_str
                        + ".jpg",
                        (disp_img * 255).astype(np.uint8),
                    )

                loss_dict = self.criterion(
                    output, img, text_input, mask_img, mask_text, use_diffusion
                )
                loss = loss_dict["loss_total"]
                loss_sum += loss.item()

                # recording the running average of the loss
                for key in loss_dict:
                    if key not in avg_loss_dict:
                        avg_loss_dict[key] = loss_dict[key].item()
                    else:
                        avg_loss_dict[key] += loss_dict[key].item()

        for key in avg_loss_dict:
            if key == "loss_total":
                avg_loss_dict[key] /= len(val_loader)
            else:
                avg_loss_dict[key] /= (
                    len(val_loader) / 3
                )  # each loss only occurs 1/3 of the time

            # wandb.log({key + '_avg_val': avg_loss_dict[key]}, step=epoch)

        return loss_sum / len(val_loader)

    def get_text_loss(self, pred, target):
        # pred: (batch_size, seq_len, vocab_size)
        # target: (batch_size, seq_len)
        pred = pred.view(-1, pred.size(-1))  # (batch_size*seq_len, vocab_size)
        target = target.view(-1)  # (batch_size*seq_len)

        # cross entropy loss, ignoring padding
        loss = torch.nn.functional.cross_entropy(pred, target, ignore_index=0)

        return loss

    def get_diffusion_loss(self, pred_noise, noise):
        return torch.nn.functional.mse_loss(pred_noise, noise)

    def criterion(self, output, img, text_input, mask_img, mask_text, use_diffusion):
        text_loss = self.get_text_loss(output["pred_text"], text_input["input_ids"])

        # applying unit gaussian prior to combined image and text features
        combined_prior_mean = torch.zeros_like(output["combined_embedding_means"]).to(
            self.device
        )
        combined_prior_logvar = torch.zeros_like(
            output["combined_embedding_logvars"]
        ).to(self.device)
        combined_kl_loss = kl_divergence(
            output["combined_embedding_means"],
            output["combined_embedding_logvars"],
            combined_prior_mean,
            combined_prior_logvar,
        )

        # if hasattr(config, 'DIFFUSION') and config.DIFFUSION:
        if use_diffusion:
            img_loss_diffusion = self.get_diffusion_loss(
                output["pred_img_noise"], output["gt_img_noise"]
            )
            # img_loss_L2 = torch.nn.functional.mse_loss(output['pred_img'], img)
            img_loss_L2 = torch.tensor(0).to(self.device)
            img_loss_gaussian = torch.tensor(0).to(self.device)
        else:
            # applying gaussian loss between output['pred_img'] and img (t2i is not deterministic, so using max likelihood)
            img_loss_gaussian = torch.nn.GaussianNLLLoss()(
                output["pred_img_means"], img, torch.exp(output["pred_img_logvars"])
            )  # input, target, variance
            # applying L2 loss between output['pred_img'] and pred_img (reconstruction is deterministic)
            img_loss_L2 = torch.nn.functional.mse_loss(output["pred_img"], img)
            img_loss_diffusion = torch.tensor(0).to(self.device)

        if self.args.debug:
            print("img_loss_gaussian: ", img_loss_gaussian)
            print("img_loss_L2: ", img_loss_L2)
            print("text_loss: ", text_loss)
            print("kl_loss: ", combined_kl_loss)
            print("diffusion_loss: ", img_loss_diffusion)

        if mask_img:
            print("masking img")
            loss_total = (
                self.config.LAMBDA_TEXT * text_loss
                + self.config.LAMBDA_KL * combined_kl_loss
            )  # supervise text and kl
            # loss_total = config.LAMBDA_IMAGE * img_loss + config.LAMBDA_KL * combined_kl_loss # supervise img and kl
            # loss_total = config.LAMBDA_IMAGE * img_loss + config.LAMBDA_TEXT * text_loss + config.LAMBDA_KL * combined_kl_loss # supervise img, text, and kl

            # ensuring each loss is weighted equally
            # unnormalized_loss = abs(text_loss) + abs(combined_kl_loss)
            # text_ratio = abs(unnormalized_loss) / abs(text_loss)
            # kl_ratio = abs(unnormalized_loss) / abs(combined_kl_loss)
            # loss_total = text_ratio * config.LAMBDA_TEXT * text_loss + kl_ratio * config.LAMBDA_KL * combined_kl_loss
            # print('loss_total: ', loss_total)
            # print('unnormalized_loss: ', unnormalized_loss)
            # print('text_ratio: ', text_ratio)
            # print('kl_ratio: ', kl_ratio)
            # print('text_ratio * text_loss: ', text_ratio * text_loss)
            # print('kl_ratio * combined_kl_loss: ', kl_ratio * combined_kl_loss)

            # ensuring each loss is weighted equally
            unnormalized_loss = abs(text_loss) + abs(combined_kl_loss)
            text_ratio = abs(unnormalized_loss) / abs(text_loss)
            kl_ratio = abs(unnormalized_loss) / abs(combined_kl_loss)
            # loss_total = text_ratio * config.LAMBDA_TEXT * text_loss + kl_ratio * config.LAMBDA_KL * combined_kl_loss
            # print('loss_total: ', loss_total)
            # print('unnormalized_loss: ', unnormalized_loss)
            # print('text_ratio: ', text_ratio)
            # print('kl_ratio: ', kl_ratio)
            # print('text_ratio * text_loss: ', text_ratio * text_loss)
            # print('kl_ratio * combined_kl_loss: ', kl_ratio * combined_kl_loss)

            return {
                "loss_total": loss_total,
                "text_only_loss_total": loss_total,
                "t2t_loss": text_loss,
                "t2i_loss": img_loss_gaussian,
                "text_only_combined_kl_loss": combined_kl_loss,
                "diffusion_loss": img_loss_diffusion,
            }
        elif mask_text:
            print("masking text")
            # if hasattr(config, 'DIFFUSION') and config.DIFFUSION:
            if use_diffusion:
                loss_total = (
                    self.config.LAMBDA_IMAGE * img_loss_diffusion
                    + self.config.LAMBDA_KL * combined_kl_loss
                )  # supervise img and kl
            else:
                # loss_total = config.LAMBDA_IMAGE * img_loss_gaussian + config.LAMBDA_KL * combined_kl_loss # supervise img (Gaussian) and kl
                loss_total = (
                    self.config.LAMBDA_IMAGE * img_loss_L2
                    + self.config.LAMBDA_KL * combined_kl_loss
                )  # supervise img (L2) and kl
                # loss_total = config.LAMBDA_TEXT * text_loss + config.LAMBDA_KL * combined_kl_loss # supervise text and kl
                # loss_total = config.LAMBDA_IMAGE * img_loss + config.LAMBDA_TEXT * text_loss + config.LAMBDA_KL * combined_kl_loss # supervise img, text, and kl

            # ensuring each loss is weighted equally
            # unnormalized_loss = abs(img_loss_gaussian) + abs(combined_kl_loss)
            # img_ratio = abs(unnormalized_loss) / abs(img_loss_gaussian)
            # kl_ratio = abs(unnormalized_loss) / abs(combined_kl_loss)

            # loss_total = img_ratio * config.LAMBDA_IMAGE * img_loss_gaussian + kl_ratio * config.LAMBDA_KL * combined_kl_loss
            # print('loss_total: ', loss_total)
            # print('unnormalized_loss: ', unnormalized_loss)
            # print('img_ratio: ', img_ratio)
            # print('kl_ratio: ', kl_ratio)
            # print('img_ratio * img_loss_gaussian: ', img_ratio * img_loss_gaussian)
            # print('kl_ratio * combined_kl_loss: ', kl_ratio * combined_kl_loss)

            return {
                "loss_total": loss_total,
                "img_only_loss_total": loss_total,
                "i2i_loss": img_loss_L2,
                "i2t_loss": text_loss,
                "img_only_combined_kl_loss": combined_kl_loss,
                "diffusion_loss": img_loss_diffusion,
            }
        else:
            print("not masking")
            # if hasattr(config, 'DIFFUSION') and config.DIFFUSION:
            if use_diffusion:
                loss_total = (
                    self.config.LAMBDA_IMAGE * img_loss_diffusion
                    + self.config.LAMBDA_TEXT * text_loss
                    + self.config.LAMBDA_KL * combined_kl_loss
                )  # supervise img (diffusion), text, and kl
            else:
                loss_total = (
                    self.config.LAMBDA_IMAGE * img_loss_L2
                    + self.config.LAMBDA_TEXT * text_loss
                    + self.config.LAMBDA_KL * combined_kl_loss
                )  # supervise img, text (L2), and kl

            # ensuring each loss is weighted equally
            # unnormalized_loss = abs(img_loss_L2) + abs(text_loss) + abs(combined_kl_loss)
            # img_ratio = abs(unnormalized_loss) / abs(img_loss_L2)
            # text_ratio = abs(unnormalized_loss) / abs(text_loss)
            # kl_ratio = abs(unnormalized_loss) / abs(combined_kl_loss)
            # loss_total = img_ratio * config.LAMBDA_IMAGE * img_loss_L2 + text_ratio * config.LAMBDA_TEXT * text_loss + kl_ratio * config.LAMBDA_KL * combined_kl_loss
            # print('loss_total: ', loss_total)
            # print('unnormalized_loss: ', unnormalized_loss)
            # print('img_ratio: ', img_ratio)
            # print('text_ratio: ', text_ratio)
            # print('kl_ratio: ', kl_ratio)
            # print('img_ratio * img_loss_L2: ', img_ratio * img_loss_L2)
            # print('text_ratio * text_loss: ', text_ratio * text_loss)
            # print('kl_ratio * combined_kl_loss: ', kl_ratio * combined_kl_loss)

            return {
                "loss_total": loss_total,
                "combined_loss_total": loss_total,
                "combined_img_loss": img_loss_L2,
                "combined_text_loss": text_loss,
                "combined_kl_loss": combined_kl_loss,
                "diffusion_loss": img_loss_diffusion,
            }


if __name__ == "__main__":
    trainer = Trainer()
    print(trainer.config)
    trainer.train()
