from typing import *

import torch
import torch.nn as nn
from fairseq.modules import TransformerSentenceEncoder
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from fidameval.utils import DEVICE, Config, unpad_sequence


class ModelConfig(Config):
    encoder: str
    vocab_size: int
    nhid: int
    emb_dim: int
    is_binary: bool = True
    pad_idx: Optional[int] = None
    mask_idx: Optional[int] = None
    one_hot_embedding: bool = True
    num_layers: int = 1
    num_heads: int = 1
    learned_pos_embedding: bool = True
    non_linear_decoder: bool = False


class LanguageClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.vocab_size = config.vocab_size
        self.is_binary = config.is_binary
        self.pad_idx = config.pad_idx
        self.mask_idx = config.mask_idx
        self.one_hot_embedding = config.one_hot_embedding
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.non_linear_decoder = config.non_linear_decoder

        if config.encoder == "lstm":
            self.emb_dim = config.emb_dim
            self.positional_embeddings = None
            self.nhid = config.nhid

            self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim)
            self.encoder = nn.LSTM(
                self.emb_dim, self.nhid, self.num_layers, batch_first=True
            )
        elif config.encoder == "transformer":
            self.emb_dim = config.emb_dim * config.num_heads
            self.nhid = self.emb_dim

            self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim)
            self.encoder = TransformerSentenceEncoder(
                padding_idx=self.pad_idx,
                vocab_size=self.vocab_size,
                num_encoder_layers=config.num_layers,
                embedding_dim=self.emb_dim,
                ffn_embedding_dim=config.nhid,
                num_attention_heads=config.num_heads,
                dropout=0.0,
                attention_dropout=0.0,
                activation_dropout=0.0,
                learned_pos_embedding=config.learned_pos_embedding,
            )
            self.positional_embeddings = self.encoder.embed_positions  # copy?
            self.encoder.embed_positions = None
        else:
            raise ValueError("Encoder type must be 'lstm' or 'transformer'")

        if config.non_linear_decoder:
            # Based on BERT imp of fairseq
            non_linear_decoder = nn.Sequential(
                nn.Linear(self.nhid, self.nhid),
                nn.ReLU(),
                nn.LayerNorm(self.nhid),
            )
            self.lm_decoder = nn.Sequential(
                non_linear_decoder, nn.Linear(self.nhid, config.vocab_size)
            )
            self.binary_decoder = nn.Sequential(
                non_linear_decoder, nn.Linear(self.nhid, 1)
            )
        else:
            self.lm_decoder = nn.Linear(self.nhid, config.vocab_size)
            self.binary_decoder = nn.Linear(self.nhid, 1)

        self.init_weights()
        self.to(DEVICE)

    def init_weights(self):
        initrange = 0.1
        if self.one_hot_embedding:
            max_emb_size = max(self.vocab_size, self.emb_dim)
            self.embeddings.weight.data = torch.eye(max_emb_size)[
                : self.vocab_size, : self.emb_dim
            ]
            self.embeddings.weight.requires_grad = False
        else:
            self.embeddings.weight.data.uniform_(-initrange, initrange)

        lm_decoder = self.lm_decoder[-1] if self.non_linear_decoder else self.lm_decoder
        binary_decoder = (
            self.binary_decoder[-1] if self.non_linear_decoder else self.binary_decoder
        )

        lm_decoder.bias.data.fill_(1 / self.vocab_size)
        lm_decoder.weight.data.uniform_(-initrange, initrange)
        binary_decoder.weight.data.uniform_(-initrange, initrange)

        if self.pad_idx is not None:
            self.embeddings.weight.data[self.pad_idx] = 0.0

    def create_inputs_embeds(
        self, input_ids: Tensor, add_positional: bool = True
    ) -> Tensor:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        if (self.positional_embeddings is not None) and add_positional:
            return self.embeddings(input_ids) + self.positional_embeddings(input_ids)
        else:
            return self.embeddings(input_ids)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        input_lengths: Optional[Tensor] = None,
        mask_ids: Optional[List[List[int]]] = None,
        return_attention=False,
        return_hidden=False,
        return_hidden_only=False,
        pseudo_ll=False,
    ):
        if inputs_embeds is None and input_ids is None:
            raise ValueError("inputs_embeds or input_ids must be provided")
        if input_ids is not None and input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if inputs_embeds is None:
            inputs_embeds = self.create_inputs_embeds(input_ids.to(DEVICE))
        if inputs_embeds.ndim == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        inputs_embeds = inputs_embeds.to(DEVICE)

        if isinstance(self.encoder, nn.LSTM):
            if input_lengths is not None:
                inputs_embeds = pack_padded_sequence(
                    inputs_embeds, input_lengths, batch_first=True, enforce_sorted=False
                )

            hidden, _ = self.encoder(inputs_embeds)

            if isinstance(hidden, PackedSequence):
                hidden, _ = pad_packed_sequence(hidden, batch_first=True)
        elif pseudo_ll:
            assert input_ids is not None
            assert self.mask_idx is not None
            hidden = torch.zeros_like(inputs_embeds, device=DEVICE)

            for idx in range(inputs_embeds.shape[1]):
                masked_input = input_ids.clone()
                masked_input[:, idx] = self.mask_idx
                masked_embeds = self.create_inputs_embeds(masked_input.to(DEVICE))
                output = self.encoder(
                    masked_input,
                    token_embeddings=masked_embeds,
                    last_state_only=True,
                    attn_mask=None,
                )
                hidden[:, idx] = output[0][0].transpose(0, 1)[:, idx]
        else:
            attn_mask = None
            token_proxy = (
                torch.zeros(inputs_embeds.shape[:-1])
                if input_ids is None
                else input_ids
            )
            output = self.encoder(
                token_proxy,
                token_embeddings=inputs_embeds,
                last_state_only=True,
                attn_mask=attn_mask,
            )
            hidden = output[0][0].transpose(0, 1)  # T x B x D -> B x T x D

            if mask_ids is not None:
                hidden = torch.cat(
                    [hidden[idx, mask_idx] for idx, mask_idx in enumerate(mask_ids)]
                )

        if return_hidden_only:
            return hidden

        if self.is_binary:
            if isinstance(self.encoder, TransformerSentenceEncoder):
                final_hidden = hidden[:, 0, :]
            elif input_lengths is None:
                final_hidden = hidden[:, -1, :]
            else:
                batch_size = hidden.shape[0]
                final_hidden = hidden[range(batch_size), input_lengths - 1]

            predictions = self.binary_decoder(final_hidden)
        else:
            predictions = self.lm_decoder(hidden)

            if mask_ids is None and input_lengths is not None:
                predictions = unpad_sequence(predictions, lengths=input_lengths)

        predictions = predictions.squeeze(1)

        if return_attention:
            print("Attention maps currently not supported!")

        if return_hidden:
            return predictions, hidden

        return predictions

    @property
    def num_parameters(self):
        return sum(torch.prod(torch.tensor(x.shape)) for x in self.parameters())
