# Puzzle Mining

This repository contains code for generating synthetic reasoning data by solving puzzles and using it to fine-tune a large language model.

Following [FineMathLlama](https://huggingface.co/HuggingFaceTB/FineMath-Llama-3B), tokenization uses [datatrove](https://github.com/huggingface/datatrove) and training is done in [nanotron](https://github.com/huggingface/nanotron).

### Reproduce results

```bash
micromamba create -n puzzle_mining -f env.yaml
micromamba activate puzzle_mining
pip install flash-attn==2.7.3

python3 eval.py --checkpoint HuggingFaceTB/FineMath-Llama-3B

mkdir checkpoints; cd checkpoints
git clone https://huggingface.co/jonpai/sudoku1
cd ..
python3 eval.py --checkpoint checkpoints/sudoku1
```

### Generate puzzles

```bash
python3 sudoku.py --puzzle-spec 4,100,2 --puzzle-spec 4,500,3 --puzzle-spec 4,1000,4 --puzzle-spec 9,10000,5 --puzzle-spec 9,50000,9 --puzzle-spec 9,50000,12
```

### Train model

```bash
git clone git@github.com:huggingface/nanotron.git
cd nanotron
pip install -e .

cd ..
git clone https://huggingface.co/HuggingFaceTB/FineMath-Llama-3B

cd nanotron
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 run_train.py --config-file ../puzzle_mining/train_config.yaml
```
