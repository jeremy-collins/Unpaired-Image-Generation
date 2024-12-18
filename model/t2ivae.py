import torch
import torch.nn as nn
import timm
from transformers import T5Model, T5EncoderModel, T5Tokenizer, BertModel, BertTokenizer, T5ForConditionalGeneration
from model.pl_resnet_ae import * # resnet encoder and decoder
import numpy as np
from model.config_utils import parse_config_args
# from model.t2idiffusion import Diffusion
from model.diffusers_utils import Diffusion
from model.unet_utils import *

class T2IVAE(nn.Module):
    def __init__(self):
        super(T2IVAE, self).__init__()
        self.config, self.args = parse_config_args()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.img_size = self.get_img_size()
        self.build_model()

    def get_img_size(self):
        if self.config.DATASET == 'coco':
            img_size = 224
        elif self.config.DATASET == 'cifar100':
            img_size = 32
        elif self.config.DATASET == 'cifar10':
            img_size = 32
        else:
            raise ValueError('invalid dataset')
    
        return img_size

    def build_model(self):
        if not hasattr(self.config, 'ARCHITECTURE') or self.config.ARCHITECTURE == 'resnet18':
            self.img_encoder = get_resnet18_encoder(first_conv=False, maxpool1=False).to(self.device)
            self.gaussian_img_decoder = get_resnet18_decoder(latent_dim=self.config.LATENT_DIM, input_height=self.img_size, first_conv=False, maxpool1=False, nc=6).to(self.device)
        elif self.config.ARCHITECTURE == 'resnet50':
            self.img_encoder = get_resnet50_encoder(first_conv=False, maxpool1=False).to(self.device)
            self.gaussian_img_decoder = get_resnet50_decoder(latent_dim=self.config.LATENT_DIM, input_height=self.img_size, first_conv=False, maxpool1=False, nc=6).to(self.device)
        elif self.config.ARCHITECTURE == 'resnet101':
            self.img_encoder = get_resnet101_encoder(first_conv=False, maxpool1=False).to(self.device)
            self.gaussian_img_decoder = get_resnet101_decoder(latent_dim=self.config.LATENT_DIM, input_height=self.img_size, first_conv=False, maxpool1=False, nc=6).to(self.device)
        elif self.config.ARCHITECTURE == 'resnet152':
            self.img_encoder = get_resnet152_encoder(first_conv=False, maxpool1=False).to(self.device)
            self.gaussian_img_decoder = get_resnet152_decoder(latent_dim=self.config.LATENT_DIM, input_height=self.img_size, first_conv=False, maxpool1=False, nc=6).to(self.device)
        else:
            raise ValueError('invalid architecture')
        
        if hasattr(self.config, 'PRETRAINED_IMG_ENC') and self.config.PRETRAINED_IMG_ENC is not None:
            img_enc_path = 'checkpoints/' + self.config.PRETRAINED_IMG_ENC + '.pt'
            self.img_encoder.load_state_dict(torch.load(img_enc_path), strict=False)
            print('loaded pretrained img encoder from', img_enc_path)

        t5_size = 'small'
        self.t5 = T5ForConditionalGeneration.from_pretrained('t5-' + t5_size).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-' + t5_size)

        self.combined_mlp = nn.Sequential(
            nn.Linear(self.config.LATENT_DIM * 2, self.config.LATENT_DIM * 2),
            nn.ReLU(),
            nn.Linear(self.config.LATENT_DIM * 2, self.config.LATENT_DIM),
            nn.ReLU(),
        )

        self.text_decoder_proj = nn.LazyLinear(512 * self.config.MAX_SEQ_LEN)
        self.img_feat_proj = nn.LazyLinear(self.config.LATENT_DIM)
        self.text_feat_proj = nn.LazyLinear(self.config.LATENT_DIM)
        self.combined_mean_proj = nn.LazyLinear(self.config.LATENT_DIM)
        self.combined_logvar_proj = nn.LazyLinear(self.config.LATENT_DIM)

        if hasattr(self.config, 'FREEZE_IMG_ENC') and self.config.FREEZE_IMG_ENC:
            for param in self.img_encoder.parameters():
                param.requires_grad = False
            # freezing the projections too
            for param in self.img_feat_proj.parameters():
                param.requires_grad = False
            for param in self.combined_mean_proj.parameters():
                param.requires_grad = False
            for param in self.combined_logvar_proj.parameters():
                param.requires_grad = False
            for param in self.combined_mlp.parameters():
                param.requires_grad = False

            print('frozen img encoder and projections')

        if hasattr(self.config, 'DIFFUSION') and self.config.DIFFUSION:
            del self.gaussian_img_decoder
            self.gaussian_img_decoder = None
            self.diffusion = Diffusion(img_size=self.img_size, noise_steps=1000)
            self.unet = UNet_conditional(num_classes=self.config.LATENT_DIM,time_dim=self.config.LATENT_DIM).to(self.device)
        else:
            self.diffusion = None
            self.unet = None

    def get_combined_embedding(self, img_feats, text_feats):
        if not hasattr(self.config, 'CONDITIONING') or self.config.CONDITIONING == 'concat-mlp':
            concat_embeddings = torch.cat((img_feats, text_feats), dim=1)
            combined_embeddings = self.combined_mlp(concat_embeddings)
            combined_means = self.combined_mean_proj(combined_embeddings)
            combined_logvars = self.combined_logvar_proj(combined_embeddings) # TODO: try fixing this to a constant (eg -4) for testing
            # combined_logvars = -4 * torch.ones_like(combined_means)
        return combined_means, combined_logvars
    
    def sample_gaussian(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, img, text_inputs, mask_img=False, mask_text=False, use_diffusion=False):
        text = text_inputs["input_ids"].to(self.device) # token ids (batch_size, seq_len)

        img_feats = self.img_encoder(img)

        # creating placeholder for text decoder
        # start token at the first position and pad tokens elsewhere
        self.placeholder_input = torch.full((text.shape[0], self.config.MAX_SEQ_LEN), self.tokenizer.pad_token_id).to(self.device) # to address end of batch error
        self.placeholder_input[:, 0] = self.tokenizer.eos_token_id # set first token to be start token

        # text encoder
        text_encoder_out = self.t5.encoder(input_ids=text, attention_mask=text_inputs["attention_mask"].to(self.device))
        text_feats = text_encoder_out.last_hidden_state

        # flatten and project image features into mean and logvar
        flattened_img_feats = img_feats.view(img_feats.shape[0], -1).to(self.device)
        img_feat_proj = self.img_feat_proj(flattened_img_feats)

        # flatten and project text features into mean and logvar
        flattened_text_feats = text_feats.view(text_feats.shape[0], -1).to(self.device)
        text_feat_proj = self.text_feat_proj(flattened_text_feats)

        # combined embeddings
        if self.args.sample_latent:
            # sampling from unit gaussian
            self.combined_embedding = torch.randn_like(img_feat_proj)
            combined_embedding_means = torch.zeros_like(img_feat_proj)
            combined_embedding_logvars = torch.zeros_like(img_feat_proj)
        else:
            # if mask_img:
            #     # replace img features with zeros
            #     combined_embedding_means, combined_embedding_logvars  = self.get_combined_embedding(torch.zeros_like(img_feat_proj), text_feat_proj)
            # elif mask_text:
            #     # replace text features with zeros
            #     combined_embedding_means, combined_embedding_logvars  = self.get_combined_embedding(img_feat_proj, torch.zeros_like(text_feat_proj))
            # else:
            #   combined_embedding_means, combined_embedding_logvars  = self.get_combined_embedding(img_feat_proj, text_feat_proj)

            # we now want to mask out instances instead of entire batches
            # both projs have shape (batch_size, latent_dim)
            # mask_img and mask_text are boolean tensors of shape (batch_size,)
            img_feat_proj = img_feat_proj * (1 - mask_img[:, None])
            text_feat_proj = text_feat_proj * (1 - mask_text[:, None])
            combined_embedding_means, combined_embedding_logvars  = self.get_combined_embedding(img_feat_proj, text_feat_proj)

            # combined_embedding_logvars -= 4 # lowering logvars by subtracting a constant (e.g. 3-5ish)
            self.combined_embedding = self.sample_gaussian(combined_embedding_means, combined_embedding_logvars) 
            # self.combined_embedding = combined_embedding_means # TODO: REMOVE THIS LINE
        
        # img decoder
        # if hasattr(self.config, 'DIFFUSION') and self.config.DIFFUSION:
        if use_diffusion:
            pred_img_means = None
            pred_img_logvars = None
            t = self.diffusion.sample_timesteps(img.shape[0]).to(self.device)
            # noise_step = torch.randint(0, self.diffusion.noise_steps, (img.shape[0],)).to(self.device)
            noised_img, noise = self.diffusion.noise_images(img, t)
            
            label = self.combined_embedding

            # if np.random.random() < 0.1:
            #     label = None # for classifer-free guidance
            #     unconditioned = True
            # else:
            unconditioned = False
                
            pred_noise = self.unet(noised_img, t, label)
            pred_img = noised_img - pred_noise
            # pred_img = noised_img + pred_noise

        else:
            pred_img_gaussian = self.gaussian_img_decoder(self.combined_embedding)
            pred_img_means = pred_img_gaussian[:, :3, :, :]
            pred_img_logvars = pred_img_gaussian[:, 3:, :, :] # lowering logvars by subtracting a constant (e.g. 3-5ish)
            pred_img = self.sample_gaussian(pred_img_means, pred_img_logvars) # - 5)
            pred_noise = torch.zeros_like(pred_img)
            noise = torch.zeros_like(pred_img)
            t = torch.zeros(img.shape[0]).to(self.device)
            unconditioned = False
            
        # text decoder
        # text_decoder_input = self.text_decoder_proj(sampled_text_latent).view(-1, self.config.MAX_SEQ_LEN, 512)
        text_decoder_input = self.text_decoder_proj(self.combined_embedding).view(-1, self.config.MAX_SEQ_LEN, 512)
        text_decoder_out = self.t5.decoder(input_ids=self.placeholder_input, encoder_hidden_states=text_decoder_input, encoder_attention_mask=text_inputs["attention_mask"].to(self.device))
        pred_text = self.t5.lm_head(text_decoder_out.last_hidden_state) # (batch_size, seq_len, vocab_size)
        pred_text = pred_text.view(-1, text.shape[1], self.t5.config.vocab_size)

        # text to image
        # t2i_input = self.img_decoder_proj(sampled_text_latent).view(-1, 512, self.img_size // 32, self.img_size // 32)
        # t2i_input = self.img_decoder_proj(combined_embedding).view(-1, 512, self.img_size // 32, self.img_size // 32)
        # pred_img_t2i = self.img_decoder(t2i_input)
        pred_img_t2i = pred_img_means
        # pred_img_t2i_gaussian = self.gaussian_img_decoder(t2i_input)
        # pred_img_t2i_means = pred_img_t2i_gaussian[:, :3, :, :]
        # pred_img_t2i_logvars = pred_img_t2i_gaussian[:, 3:, :, :]

        # image to text
        # i2t_input = self.text_decoder_proj(sampled_img_latent).view(-1, self.config.MAX_SEQ_LEN, 512)
        # i2t_input = self.text_decoder_proj(combined_embedding).view(-1, self.config.MAX_SEQ_LEN, 512)
        # i2t_decoder_out = self.t5.decoder(input_ids=self.placeholder_input, encoder_hidden_states=i2t_input, encoder_attention_mask=text_inputs["attention_mask"].to(device))
        # pred_text_i2t = self.t5.lm_head(i2t_decoder_out.last_hidden_state) # (batch_size, seq_len, vocab_size)
        # pred_text_i2t = pred_text_i2t.view(-1, text.shape[1], self.t5.config.vocab_size)
        pred_text_i2t = pred_text

        return {
            "pred_img": pred_img,
            "pred_img_means": pred_img_means,
            "pred_img_logvars": pred_img_logvars,
            "pred_img_noise": pred_noise,
            "gt_img_noise": noise,
            "noise_step": t,
            "pred_text": pred_text,
            "img_feats": img_feats,
            "text_feats": text_feats,
            "img_feat_proj": img_feat_proj,
            "text_feat_proj": text_feat_proj,
            "combined_embedding": self.combined_embedding,

            "combined_embedding_means": combined_embedding_means,
            "combined_embedding_logvars": combined_embedding_logvars,
            "unconditioned": unconditioned,
        }
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T2IVAE().to(device)
    img = torch.randn(1, 3, 224, 224).to(device)
    text = "This is a sentence."
    text = model.tokenizer(text, return_tensors="pt", padding=True)
    output = model(img, text)

    print("pred_img", output["pred_img"].shape)
    print("pred_text", output["pred_text"].shape)
    print("img_feats", output["img_feats"].shape)
    print("text_feats", output["text_feats"].shape)
    print("img_feat_proj", output["img_feat_proj"].shape)
    print("text_feat_proj", output["text_feat_proj"].shape)
    print("combined_embedding_means", output["combined_embedding_means"].shape)
    print("combined_embedding_logvars", output["combined_embedding_logvars"].shape)

    decoded_text = model.tokenizer.batch_decode(output["pred_text"], skip_special_tokens=True)
    print("decoded pred_text: ", decoded_text)