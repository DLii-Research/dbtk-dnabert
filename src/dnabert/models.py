from dbtk._utils import export
from dbtk.nn.models import DbtkModel
from dbtk.nn import layers
from deprecated import deprecated
import lightning as L
from transformers import PretrainedConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union

from .datamodules import DnaBertPretrainingDataModule
from .tokenizers import DnaTokenizer

@export
class DnaBert(DbtkModel):
    class Config(PretrainedConfig):
        def __init__(
            self,
            kmer=3,
            kmer_stride=1,
            normalize_sequences=True,
            embed_dim: int = 64,
            num_heads: int = 8,
            num_layers: int = 8,
            feedforward_dim: int = 256,
            activation: str = "gelu",
            max_length: int = 1024,
            **kwargs
        ):
            super().__init__(**kwargs)
            self.kmer = kmer
            self.kmer_stride = kmer_stride
            self.normalize_sequences = normalize_sequences
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.num_layers = num_layers
            self.feedforward_dim = feedforward_dim
            self.activation = activation
            self.max_length = max_length

    config_class = Config

    def __init__(self, config: Optional[Union[Config, dict]] = None):
        super().__init__(config)

        self.tokenizer = DnaTokenizer(
            kmer=self.config.kmer,
            kmer_stride=self.config.kmer_stride,
            normalize_sequences=self.config.normalize_sequences
        )

        if isinstance(self.config.activation, str):
            activation = getattr(F, self.config.activation)
        else:
            activation = self.config.activation

        self.transformer = layers.TransformerEncoder(
            layers.TransformerEncoderBlock(
                mha=layers.RelativeMultiHeadAttention(
                    embed_dim=self.config.embed_dim,
                    num_heads=self.config.num_heads,
                    max_length=self.config.max_length
                ),
                feedforward_dim=self.config.feedforward_dim,
                feedforward_activation=activation
            ),
            num_layers=self.config.num_layers
        )

        self.embeddings = nn.Embedding(
            len(self.tokenizer),
            self.config.embed_dim,
            padding_idx=self.tokenizer.vocab["[PAD]"]
        )

    def forward(
        self,
        kmers: torch.Tensor
    ):
        # Prepend class token and append separator token
        kmers = F.pad(kmers, (1, 0), mode="constant", value=self.tokenizer.vocab["[CLS]"])
        tokens = self.embeddings(kmers)

        # Pass through transformer
        output = self.transformer(tokens)

        # Separate embeddings
        transformed_class_tokens = output[:, 0]
        transformed_kmers = output[:, 1:-1]

        return {
            "class": transformed_class_tokens,
            "tokens": transformed_kmers
        }

@export
class DnaBertForPretraining(DbtkModel):
    class Config(PretrainedConfig):
        is_composition = True
        def __init__(
            self,
            base: Optional[Union[DnaBert, DnaBert.Config, dict]] = None,
            min_mask_ratio: float = 0.15,
            max_mask_ratio: float = 0.15,
            **kwargs
        ):
            super().__init__(**kwargs)
            self.base = base if base is not None else DnaBert.Config()
            self.min_mask_ratio = min_mask_ratio
            self.max_mask_ratio = max_mask_ratio

    config_class = Config
    base_model_prefix = "base"

    def __init__(self, config: Optional[Union[Config, dict]] = None):
        super().__init__(config)

        # Setup base model
        self.base = self.instantiate_model("base", DnaBert)
        self.mask_head = nn.Linear(self.base.config.embed_dim, self.base.tokenizer.num_token_ids)

    def _apply_random_masking(self, kmers: torch.Tensor, inplace: bool = False):
        """
        Randomly replace contiguous blocks of kmers with mask tokens.
        """
        # Compute mask regions
        lengths = torch.sum(kmers != 0, dim=1)
        mask_ratios = self.config.min_mask_ratio + torch.rand(lengths.shape[0], device=kmers.device)*(self.config.max_mask_ratio - self.config.min_mask_ratio)
        num_mask_tokens = torch.clamp(torch.round(lengths * mask_ratios).long(), min=1)
        offsets = torch.rand(size=(lengths.shape[0],), device=kmers.device)
        offsets = torch.round(offsets * (lengths - num_mask_tokens)).long()

        # Compute mask
        indices = torch.arange(kmers.shape[-1], device=kmers.device).expand(lengths.shape[0], -1)
        mask = (offsets.unsqueeze(-1) <= indices) & ((offsets+num_mask_tokens).unsqueeze(-1) > indices)

        # Apply masking
        if not inplace:
            kmers = kmers.clone()
        targets = kmers[mask]
        kmers[mask] = self.base.tokenizer.vocab["[MASK]"]
        return kmers, mask, targets

    def forward(self, kmers: torch.Tensor):
        kmers, mask, targets = self._apply_random_masking(kmers, inplace=True)
        output = self.base(kmers)["tokens"]
        masked_output = output[mask]
        return self.mask_head(masked_output), targets

    def _step(self, mode: str, batch: Dict[str, torch.Tensor]):
        kmers = batch["kmers"]
        output, targets = self(kmers)
        loss = F.cross_entropy(output, targets)
        accuracy = (output.argmax(dim=-1) == targets).float().mean()
        self.log(f"{mode}/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{mode}/accuracy", accuracy, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch):
        return self._step("train", batch)

    def validation_step(self, batch):
        return self._step("val", batch)

    def test_step(self, batch):
        return self._step("test", batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


@export
class DnaBertForSequenceEmbedding(DbtkModel):
    class Config(PretrainedConfig):
        def __init__(
            self,
            base: Optional[Union[DnaBert, DnaBert.Config, dict]] = None,
            **kwargs
        ):
            super().__init__(**kwargs)
            self.base = base if base is not None else DnaBert.Config()

    config_class = Config
    base_model_prefix = "base"

    def __init__(self, config: Optional[Union[Config, dict]] = None):
        super().__init__(config)
        self.base = self.instantiate_model("base", DnaBert)

    def forward(self, kmers: torch.Tensor):
        return self.base(kmers)["class"]

# Legacy

from dbtk._utils import export
from dbtk.data import tokenizers, vocabularies
from dbtk.nn import layers
from dbtk.nn.models import bert

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional

@export
class DnaBertModel(L.LightningModule):
    def __init__(
        self,
        transformer_encoder: layers.TransformerEncoder,
        kmer: int = 1,
        kmer_stride: int = 1
    ):
        super().__init__()
        self.transformer_encoder = transformer_encoder
        self.kmer = kmer
        self.kmer_stride = kmer_stride
        self.tokenizer=tokenizers.DnaTokenizer(kmer, kmer_stride)
        self.vocabulary=bert.BertVocabulary(vocabularies.dna(kmer))
        self.token_embeddings = nn.Embedding(
            num_embeddings=len(self.vocabulary),
            embedding_dim=self.embed_dim,
            padding_idx=0)
        self.return_class_embeddings = True
        self.return_item_embeddings = True

    def forward(
        self,
        src: torch.Tensor,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False,
        **kwargs
    ):
        # Construct input
        src = F.pad(src, (1, 0), mode="constant", value=self.vocabulary["[CLS]"])
        mask = (src == self.vocabulary["[PAD]"])
        src = self.token_embeddings(src)

        # Pass through transformer encoder
        output = self.transformer_encoder(
            src,
            src_key_padding_mask=mask,
            average_attention_weights=average_attention_weights,
            return_attention_weights=return_attention_weights,
            **kwargs)

        # Construct outputs
        if isinstance(output, tuple):
            output, *extra = output
        else:
            extra = ()
        # (class_tokens, output_tokens, extra)
        # (class_tokens, (output_tokens_a, output_tokens_b), extra)
        result = ()
        if self.return_class_embeddings:
            result += (output.select(-2, 0),)
        if self.return_item_embeddings:
            result += (output.narrow(-2, 1, src.shape[-2] - 1),)
        result += extra
        if len(result) == 1:
            return result[0]
        return result

    @property
    def embed_dim(self):
        return self.transformer_encoder.embed_dim

    def outputs(self, class_embeddings: Optional[bool] = None, item_embeddings: Optional[bool] = None) -> "BertModel":
        """
        Configure what the model should return.
        """
        if class_embeddings is not None:
            self.return_class_embeddings = class_embeddings
        if item_embeddings is not None:
            self.return_item_embeddings = item_embeddings
        return self


@export
class DnaBertPretrainingModel(L.LightningModule):
    def __init__(self, base: DnaBertModel):
        super().__init__()
        self.base = base
        self.predict_tokens = nn.Linear(
            self.embed_dim,
            len(self.base.vocabulary))

    def forward(
        self,
        src: torch.Tensor
    ):
        return self.base(src)

    def _step(self, mode, batch):
        src, masked_tokens = batch
        class_tokens, output = self(src)

        indices = torch.where(src.flatten() == self.base.vocabulary["[MASK]"])
        predicted = self.predict_tokens(output.flatten(0, -2)[indices])
        loss = F.cross_entropy(predicted, masked_tokens)
        num_correct = torch.sum(torch.argmax(predicted, dim=-1) == masked_tokens)

        n = masked_tokens.shape[-1]
        self.log(f"{mode}/loss", loss, prog_bar=True)
        self.log(f"{mode}/reconstruction_accuracy", num_correct.float() / n, prog_bar=True)
        return loss

    def training_step(self, batch):
        return self._step("train", batch)

    def validation_step(self, batch):
        return self._step("val", batch)

    def test_step(self, batch):
        return self._step("test", batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4) # type: ignore
        # return schedulefree.AdamWScheduleFree(self.parameters(), lr=1e-4)

    @property
    def embed_dim(self):
        return self.base.embed_dim
