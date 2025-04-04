# dbtk-dnabert

An implementation of [DNABERT](https://academic.oup.com/bioinformatics/article/37/15/2112/6128680) using Pytorch and the [deepbio-toolkit](https://pypi.org/project/deepbio-toolkit/) library.

## Getting Started

1. Install the dbtk-dnabert package
```bash
pip install dbtk-dnabert
```

2. Pull pre-trained DNABERT model
```py
from dnabert import DnaBert

# Load the pre-trained model
model = DnaBert.from_pretrained("SirDavidLudwig/dnabert", revision="64d-silva16s-250bp")
```

## Examples

Embed DNA sequences
```py
# Sequences to embed
sequences = [
    "ACTGAATGAGAC",
    "TTGAGTAGCCAA"
]

# Tokenize sequences
sequence_tokens = torch.tensor([model.tokenizer(sequence) for sequence in sequences])

# Embed sequences
output = model(sequence_tokens)

# Sequence-level embeddings from class token
embeddings = output["class"]

# Sequence-level embeddings from averaged tokens
embeddings = output["tokens"].mean(dim=1)
```

## Pre-trained Models

| Model Name | Embedding Dim. | Maximum Length | Pre-training Dataset |
| --- | --- | --- | --- |
| `64d-silva16s-250bp` | 64 | 250bp | Silva 16S |
| `768d-silva16s-250bp` | 768 | 250bp | Silva 16S |

## Development

### 1. Model Configuration

Template model configurations can be generated using the `dbtk model config` command.

### 2. Pre-training

The model can be pre-trained using the supplied configurations with the command:

```bash
dbtk model fit \
    -c ./configs/datamodules/pretrain_silva_16s_250bp.yaml \
    -c ./configs/models/pretrain_dnabert_768d_250bp.yaml \
    -c ./configs/trainers/pretrainer.yaml \
    ./logs/dnabert_768d_250bp
```

### 3. Exporting

The trained model can be exported to a Huggingface model with the following command.

```bash
dbtk model export ./logs/dnabert_768d_250bp/last.ckpt ./exports/dnabert_768d_250bp
```
