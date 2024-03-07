import math
from types import SimpleNamespace

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# from utils.config_utils import parse_config_args
from pl_resnet_ae import *  # resnet encoder and decoder
from transformers import (
    BertModel,
    BertTokenizer,
    T5EncoderModel,
    T5ForConditionalGeneration,
    T5Model,
    T5Tokenizer,
)

"""
# import xformers
from xformers.components.attention import ScaledDotProduct
from xformers.components.feedforward import build_feedforward
from xformers.components.positional_embedding import SinePositionalEmbedding
from xformers.factory import xFormer, xFormerConfig, xFormerDecoderConfig

# not working on mac for some reason
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, input_dim=4, max_len=10, num_heads=1, num_layers=1):
        super(DecoderOnlyTransformer, self).__init__()

        # Define the Feedforward network configuration
        ff_config = {
            "name": "MLP",
            "dim_model": input_dim,
            "activation": "relu",
            "dropout": 0.1,
            "hidden_layer_multiplier": 1,
        }

        # Define the Scaled Dot Product Attention configuration
        attn_config = {
            "name": "ScaledDotProduct",
            "dropout": 0.1,
            "causal": True,
            "seq_len": max_len,
        }

        # Define the Decoder Layer directly within the Decoder configuration to avoid using xFormerDecoderLayerConfig
        decoder_config = xFormerDecoderConfig(
            num_layers=num_layers,
            dim_model=input_dim,
            feedforward_config=build_feedforward(ff_config),
            attention_config=ScaledDotProduct(**attn_config),
            num_heads=num_heads,
        )

        # Define the Transformer configuration
        transformer_config = xFormerConfig(decoder_config=decoder_config)

        # Build the Transformer
        self.pos_enc = SinePositionalEmbedding(input_dim)
        self.decoder_only_transformer = xFormer.from_config(transformer_config)

    def forward(self, e_tokens):
        # Assume embedded_tokens is of shape (b=5, max_len, 512)
        e_tokens = e_tokens + self.pos_enc
        print(e_tokens)
        # return self.decoder_only_transformer(e_tokens)
        # return output
"""


class ImageTokenizer(nn.Module):
    def __init__(self, latent_dim=256, img_size=224, vocab_size=512):
        super(ImageTokenizer, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size

        # replace with different pretrained model later
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            # nn.ReLU(inplace=True),
        )

        self.deconv_blocks = nn.Sequential(
            nn.ConvTranspose2d(
                4, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                64, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                64, 3, kernel_size=7, stride=2, padding=3, output_padding=1
            ),
        )

        # can order them later s.t index 0 is the lowest value, index n-1 is the highest value
        self.vocab = torch.randn(vocab_size)
        self.vocab_embeddings = nn.Embedding(vocab_size, latent_dim)

    def encode(self, img):
        return self.conv_blocks(img)

        # normal distro with mean 0 and st dev 1 (after removing relu)
        # import matplotlib.pyplot as plt

        # plt.hist(x.flatten().detach().numpy(), bins=100, range=(-5, 5))
        # plt.savefig("img_hist.png")

        # flattened_x = x.flatten().detach().numpy()
        # import numpy as np

        # mean = np.mean(flattened_x)
        # stdev = np.std(flattened_x)

        # print(mean, stdev)

    def forward(self):
        pass

    def tokenize(self, enc_img):
        b, c, h, w = enc_img.size()
        flattened_img = enc_img.flatten(start_dim=0)

        # double check if this preserves the batch independence
        distances = torch.cdist(
            flattened_img.unsqueeze(1).unsqueeze(0),
            self.vocab.unsqueeze(0).unsqueeze(-1),
        )

        _, ids = torch.min(distances, dim=2)

        # ids: b, max_len-1
        return ids.reshape(b, -1)

    def embed(self, img_tokens):
        # b, max_len-1, d_k
        return self.vocab_embeddings(img_tokens)

    def detokenize(self, token_ids):
        # b, seq_len
        return self.vocab[token_ids]

    def decode(self, tokens):
        b = tokens.shape[0]
        tokens = tokens.view(b, 4, 28, 28)
        return self.deconv_blocks(tokens)


# modded from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# for batch first
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Changed from pe = pe.unsqueeze(1) to adding an extra dimension for batch during forward
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        # Expands the positional encoding to match the batch size of x, without extra memory allocation
        # The expansion happens automatically by broadcasting
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CustomDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(CustomDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        self.linear = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, tgt, tgt_mask):
        sa = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, is_causal=True)[0]
        tgt += self.dropout1(sa)
        tgt = self.norm1(tgt)

        ff = self.linear(tgt)
        tgt += self.dropout2(ff)
        tgt = self.norm2(tgt)

        return tgt


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self, input_dim, max_len, tokenizer, num_heads=4, num_layers=2, vocab_size=512
    ):
        super(DecoderOnlyTransformer, self).__init__()

        self.max_len = max_len
        self.tokenizer = tokenizer

        self.pos_enc = PositionalEncoding(input_dim, dropout=0.1, max_len=max_len)
        self.layers = nn.ModuleList(
            [CustomDecoderLayer(input_dim, num_heads) for _ in range(num_layers)]
        )
        self.input_dim = input_dim
        self.lm_head = nn.Linear(input_dim, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def process_layers(self, src, mask, last_only=False):
        src += self.pos_enc(src)
        for layer in self.layers:
            src = layer(src, mask)

        if last_only:
            src = src[:, -1, :].unsqueeze(1)
        src = self.lm_head(src)
        return src

    def sample_or_argmax(self, logits, sample=True):
        dist = torch.distributions.Categorical(logits=logits)
        if sample:
            tokens = dist.sample()
        else:
            tokens = logits.argmax(dim=-1)
        log_prob = dist.log_prob(tokens)
        return tokens, log_prob

    def inference(self, input_token_embedding):
        with torch.no_grad():
            # input_token_embedding: (b, 1, d_model)
            b_size = input_token_embedding.shape[0]

            generated_sequence = input_token_embedding

            token_ids = -torch.ones(b_size, 1)
            log_probs = torch.zeros(b_size, 1)

            for _ in range(1, self.max_len):
                mask = self.generate_square_subsequent_mask(
                    generated_sequence.shape[1]
                ).to(input_token_embedding.device)
                # b, 1, d_vocab
                logits = self.process_layers(generated_sequence, mask, True)
                next_token, log_prob = self.sample_or_argmax(
                    logits, sample=False
                )  # b, 1
                log_probs = torch.cat((log_probs, log_prob), dim=1)
                token_ids = torch.cat((token_ids, next_token), dim=1)

                next_token_embedding = self.tokenizer.vocab_embeddings(
                    next_token
                )  # b, 1, d_k
                generated_sequence = torch.cat(
                    (generated_sequence, next_token_embedding), dim=1
                )

            return token_ids[:, 1:], log_probs[:, 1:]

    def forward(self, img_tokens, context_vector):
        # img_tokens: b, max_len-1. context: b, 1, d_k

        input_tokens = torch.cat(
            (self.tokenizer.embed(img_tokens), context_vector), dim=1
        )  # b, max_len, d_k
        print(input_tokens.shape)
        mask = self.generate_square_subsequent_mask(self.max_len).to(
            input_tokens.device
        )
        logits = self.process_layers(input_tokens, mask)  # b, max_len, vocab_size
        output_tokens, log_probs = self.sample_or_argmax(logits)  # b, max_len

        logits, output_tokens, log_probs = (
            logits[:, 1:],
            output_tokens[:, 1:],
            log_probs[:, 1:],
        )

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), img_tokens.reshape(-1))

        output_img = self.tokenizer.detokenize(output_tokens)
        output_img = self.tokenizer.decode(output_img)
        return {
            "loss": loss,
            "logits": logits,
            "output_tokens": output_tokens,
            "log_probs": log_probs,
            "output_img": output_img,
        }


class T2IRegressive(nn.Module):
    def __init__(self, config):
        super(T2IRegressive, self).__init__()

        self.config = config

        self.img_encoder = get_resnet18_encoder(first_conv=False, maxpool1=False)
        self.text_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.t5 = T5ForConditionalGeneration.from_pretrained("t5-small")

        self.img_feat_proj = nn.LazyLinear(self.config.LATENT_DIM)
        self.text_feat_proj = nn.LazyLinear(self.config.LATENT_DIM)
        self.combined_mlp = nn.Sequential(
            nn.Linear(self.config.LATENT_DIM * 2, self.config.LATENT_DIM * 2),
            nn.ReLU(),
            nn.Linear(self.config.LATENT_DIM * 2, self.config.LATENT_DIM),
            nn.ReLU(),
        )

        self.combined_mean_proj = nn.LazyLinear(self.config.LATENT_DIM)
        self.combined_logvar_proj = nn.LazyLinear(self.config.LATENT_DIM)

        self.text_decoder_proj = nn.LazyLinear(512 * self.config.MAX_SEQ_LEN)

        self.image_tokenizer = ImageTokenizer(
            latent_dim=self.config.LATENT_DIM,
            img_size=224,
            vocab_size=self.config.VOCAB_SIZE,
        )
        self.image_decoder = DecoderOnlyTransformer(
            input_dim=self.config.LATENT_DIM,
            max_len=self.config.MAX_SEQ_LEN,
            tokenizer=self.image_tokenizer,
            num_heads=1,
            num_layers=1,
            vocab_size=self.config.VOCAB_SIZE,
        )

    def encode(self, img, text):
        text_input = self.text_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.MAX_SEQ_LEN,
        )

        if text_input["input_ids"].size(1) < self.config.MAX_SEQ_LEN:
            text_input["input_ids"] = torch.nn.functional.pad(
                text_input["input_ids"],
                (0, self.config.MAX_SEQ_LEN - text_input["input_ids"].shape[1]),
            )
        else:
            text_input["input_ids"] = text_input["input_ids"][
                :, : self.config.MAX_SEQ_LEN
            ]

        # Generate attention masks while ignoring pad token id
        text_input["attention_mask"] = text_input["input_ids"].ne(
            self.text_tokenizer.pad_token_id
        )

        encoded_img = self.img_encoder(img)
        encoded_text = self.t5.encoder(
            input_ids=text_input["input_ids"],
            attention_mask=text_input["attention_mask"],
        ).last_hidden_state

        # b, 512, 28, 28 : b, MAX_SEQ_LEN, 512
        return {
            "encoded_img": encoded_img,
            "encoded_text": encoded_text,
            "text_input": text_input,
        }

    def get_combined_embedding(self, img_feats, text_feats):
        concat_embeddings = torch.cat((img_feats, text_feats), dim=1)
        combined_embeddings = self.combined_mlp(concat_embeddings)
        combined_means = self.combined_mean_proj(combined_embeddings)
        combined_logvars = self.combined_logvar_proj(
            combined_embeddings
        )  # TODO: try fixing this to a constant (eg -4) for testing
        # combined_logvars = -4 * torch.ones_like(combined_means)
        return combined_means, combined_logvars

    def sample_gaussian(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def combine(self, e_img, e_txt, masks):
        # b, -1
        flat_e_img = e_img.flatten(start_dim=1)
        flat_e_txt = e_txt.flatten(start_dim=1)

        # b, latent_dim
        img_feat_proj = self.img_feat_proj(flat_e_img)
        text_feat_proj = self.text_feat_proj(flat_e_txt)

        (mask_img, mask_text) = masks
        if mask_img:
            # replace img features with zeros
            (
                combined_embedding_means,
                combined_embedding_logvars,
            ) = self.get_combined_embedding(
                torch.zeros_like(img_feat_proj), text_feat_proj
            )
        elif mask_text:
            # replace text features with zeros
            (
                combined_embedding_means,
                combined_embedding_logvars,
            ) = self.get_combined_embedding(
                img_feat_proj, torch.zeros_like(text_feat_proj)
            )
        else:
            (
                combined_embedding_means,
                combined_embedding_logvars,
            ) = self.get_combined_embedding(img_feat_proj, text_feat_proj)
        # combined_embedding_logvars -= 4 # lowering logvars by subtracting a constant (e.g. 3-5ish)
        combined_embedding = self.sample_gaussian(
            combined_embedding_means, combined_embedding_logvars
        )

        return {
            "combined_embedding": combined_embedding,
            "combined_embedding_means": combined_embedding_means,
            "combined_embedding_logvars": combined_embedding_logvars,
        }

    def decode(self, combined_embedding, text_input, img):
        b = combined_embedding.shape[0]

        placeholder_input = torch.full(
            (b, self.config.MAX_SEQ_LEN), self.text_tokenizer.pad_token_id
        )
        # eos_token_id ?
        placeholder_input[:, 0] = self.text_tokenizer.eos_token_id

        text_decoder_input = self.text_decoder_proj(combined_embedding).view(
            -1, self.config.MAX_SEQ_LEN, 512
        )
        text_decoder_out = self.t5.decoder(
            input_ids=placeholder_input,
            encoder_hidden_states=text_decoder_input,
            encoder_attention_mask=text_input["attention_mask"],
        )
        # (batch_size, seq_len, vocab_size)
        pred_text = self.t5.lm_head(text_decoder_out.last_hidden_state)
        print(text_input["input_ids"].shape, pred_text.shape)
        print(text_input["input_ids"])

        text_loss_fn = nn.CrossEntropyLoss()
        text_loss = text_loss_fn(
            pred_text.view(-1, self.t5.config.vocab_size),
            text_input["input_ids"].flatten(),
        )

        output_text = [
            self.text_tokenizer.decode(
                pred_text[i].argmax(dim=-1), skip_special_tokens=True
            )
            for i in range(b)
        ]

        image_tokens = self.image_tokenizer.tokenize(self.image_tokenizer.encode(img))
        image_output = self.image_decoder.forward(image_tokens, combined_embedding)

        response = {
            "pred_text": pred_text,
            "output_text": output_text,
            "text_loss": text_loss,
        }

        response.update(image_output)

        return response

    def forward(self, img, text, masks):
        # img (b, 3, 224, 224)
        # text (b)

        enc = self.encode(img, text)
        comb = self.combine(enc["encoded_img"], enc["encoded_text"], masks)
        dec = self.decode(comb["combined_embedding"], enc["text_input"], img)

        # add KL loss
        res = {
            "enc": enc,
            "comb": comb,
            "dec": dec,
        }

        return res

    def inference(self, context):
        # try t->i->t and i->t->i, then rl loss



if __name__ == "__main__":
    # Sample usage
    b_size = 5
    d_k = 16
    vocab_size = 25
    mlen = 51

    config = SimpleNamespace(MAX_SEQ_LEN=128, LATENT_DIM=32, VOCAB_SIZE=25)
    t2ir = T2IRegressive(config)
    t2ir.forward(
        torch.randn(2, 3, 224, 224), ["This is a cat", "This is a dog"], (True, True)
    )

    # img = cv2.imread("COCO_train2014_000000000009.jpg")
    # img = cv2.resize(img, (224, 224))
    # img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    # print(img.shape)
    # it = ImageTokenizer(latent_dim=d_k, img_size=224, vocab_size=vocab_size)
    # print(it.encode(img).shape)
    # print(it.decode(it.encode(img)).shape)

    # image_tokens = it.tokenize(it.encode(img))
    # print(image_tokens.shape)
    # print(it.embed(image_tokens).shape)
    # print(it.detokenize(image_tokens).shape)

    # it = ImageTokenizer(latent_dim=d_k, img_size=224, vocab_size=vocab_size)
    # image_tokens = torch.randint(low=0, high=vocab_size, size=(b_size, mlen - 1))
    # dot = DecoderOnlyTransformer(
    #     input_dim=d_k,
    #     max_len=mlen,
    #     tokenizer=it,
    #     num_heads=1,
    #     num_layers=1,
    #     vocab_size=vocab_size,
    # )
    # dot.forward(image_tokens, torch.randn(b_size, 1, d_k))
