# DNABERT

A complete DNABERT implementation in Pytorch using th deepbio-toolkit library.

## Model Configuration

Template model configurations can be generated using the `dbtk model config` command.

## Pre-training

The model can be pre-trained using the supplied configurations with the command:

```bash
dbtk model fit \
    -c ./configs/datamodules/pretrain_silva_16s_250bp.yaml \
    -c ./configs/models/pretrain_dnabert_768d_250bp.yaml \
    -c ./configs/trainers/pretrainer.yaml \
    ./logs/dnabert_768d_250bp
```

## Exporting

The trained model can be exported to a Huggingface model with the following command.

```bash
dbtk model export ./logs/dnabert_768d_250bp/last.ckpt ./exports/dnabert_768d_250bp
```
