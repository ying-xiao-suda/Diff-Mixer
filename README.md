# Diff-Mixer: An All-MLP Architecture for Probabilistic Spatiotemporal Imputation based on Diffusion Model

## Requirements

- Python 3.8.18
- PyTorch 2.1.2.
- NumPy
- Pandas


## Quick Start

### Train 

```bash
python exps.py \
  --config configs/bay.yaml \
  --nsample 100 \
  --testmissingratio 0.3 \
  --seed 42 \
  --dataset bay
```
### Test 

```bash
python exps.py \
  --config bay.yaml \
  --nsample 100 \
  --testmissingratio 0.05 \
  --seed 42 \
  --block \
  --dataset bay \
  --modelfolder [filename]
```

The experiment scripts are in the config folder.

- run.sh is used to run experiments on Linux. An example of usage is
```bash
bash run.sh
```
- test.bat is used to run experiments on Windows. An example of usage is
```bash
test.bat
```
