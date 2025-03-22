# DNABERT

A complete DNABERT implementation in Pytorch using th deepbio-toolkit library.

## Pre-training

```bash
dbtk model fit -c configs/768d.yaml ./logs/768d
```

## Exporting

```bash
dbtk model export ./logs/768d ./dist/768d
```
