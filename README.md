# training-free-hmc

Code for the paper "Training-free and consistent hierarchical multi-label classification with tabular foundation models".

## Setup

```bash
pip install -e ".[test]"
pytest
```

TabPFN downloads its model weights on first use; see the TabPFN documentation for access and licence terms. The
TabPFN scores in the paper were computed on NVIDIA RTX 4090 GPUs; on a CPU the scores can differ slightly.

## Data

The 20 benchmark datasets are the ones distributed with C-HMCNN:

```bash
git clone https://github.com/EGiunchiglia/C-HMCNN
git -C C-HMCNN checkout e887855
cp -r C-HMCNN/HMC_data data
```

## Reproducing the results

```bash
tfhmc fit --scores scores
tfhmc evaluate --scores scores --out results
tfhmc lgbm --out results/lgbm.csv
bash baselines/run.sh
tfhmc reruns --chmcnn baselines/out/chmcnn --spur baselines/out/spur --out results/reruns.csv
```

`fit` takes `--datasets` and `--shard i/k`; shards of the same dataset can run on separate GPUs and are merged by
`evaluate`. The baselines need the dependencies of the C-HMCNN and SPUR repositories and a GPU.

## Licence

MIT
