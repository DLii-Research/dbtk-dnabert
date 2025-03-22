# DNABERT

A complete DNABERT implementation in Pytorch using th deepbio-toolkit library.

## Model Configuration

Template model configurations can be generated using the `dbtk model config` command.

## Pre-training

The model can be pre-trained using the supplied configurations with the command:

```bash
dbtk model fit -c configs/768d.yaml ./logs/768d
```

## Exporting

The trained model can be exported to a Huggingface model with the following command.

```bash
dbtk model export ./logs/768d/last.ckpt ./exports/768d
```
