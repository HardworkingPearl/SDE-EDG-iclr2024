# Latent Trajectory Learning for Limited Timestamps under Distribution Shift over Time

[__[Paper]__](https://openreview.net/pdf?id=bTMMNT7IdW) 
&nbsp; 
This is the authors' official PyTorch implementation for SDE-EDG method in the **ICLR 2024 (Oral)** paper [Latent Trajectory Learning for Limited Timestamps under Distribution Shift over Time](https://openreview.net/pdf?id=bTMMNT7IdW).


## Prerequisites
- PyTorch >= 1.12.1 (with suitable CUDA and CuDNN version)
- torchvision >= 0.10.0
- Python3
- Numpy
- pandas

## Dataset
- Circle/Sine/RMNIST/Portraits/Caltran/PowerSupply download through [here].(https://github.com/WonderSeven/LSSAE?tab=readme-ov-file)
- [Ocular Disease](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k/data)

## Training
Rotated Gaussian experiment
```
python main.py --gpu_ids 0 --data_name RotatedGaussian --data_path '/dataset/Rotated_Gaussian/rotated-gaussian.pkl' --num_classes 2 --data_size '[1,2]' --source-domains 22 --intermediate-domains 3 --target-domains 5 --mode train --model-func Toy_Linear_FE --feature-dim 32 --epochs 80 --iterations 50 --train_batch_size 64 --eval_batch_size 50 --test_epoch -1 --algorithm SDE --seed 0 --save_path './logs/' --record --mlp-depth 2 --mlp-width 32

```

Rotated MNIST experiment
```

python main.py --gpu_ids 0 --data_name RMNIST --data_path '/dataset/' --num_classes 10 --data_size '[1, 28, 28]' --source-domains 10 --intermediate-domains 3 --target-domains 6 --mode train --model-func MNIST_CNN --feature-dim 512 --epochs 50 --iterations 100 --train_batch_size 48 --eval_batch_size 48 --test_epoch -1 --algorithm SDE --seed 0 --save_path './logs/' --record --mlp-depth 2 --mlp-width 512
```

Caltran experiment
'''
python main.py --gpu_ids 0 --data_name CalTran --data_path '/dataset/CalTran/' --num_classes 2 --data_size '[3, 84, 84]' --source-domains 19 --intermediate-domains 5 --target-domains 10 --mode train --model-func resnet18 --feature-dim 512 --epochs 80 --iterations 100 --train_batch_size 24 --eval_batch_size 24 --test_epoch -1 --algorithm SDE --seed 0 --save_path './logs/' --record --mlp-depth 3 --mlp-width 512
'''

Portraits experiment
'''
python main.py --gpu_ids 0 --data_name Portraits --data_path '/dataset/Portraits/' --num_classes 2 --data_size '[1, 84, 84]' --source-domains 19 --intermediate-domains 5 --target-domains 10 --mode train --model-func resnet18 --feature-dim 512 --epochs 100 --iterations 100 --train_batch_size 24 --eval_batch_size 24 --test_epoch -1 --algorithm SDE --seed 0 --save_path './logs/' --record --mlp-depth 3 --mlp-width 512
'''

Ocular Disease experiment
'''
python main.py --gpu_ids 0 --data_name Ocular --data_path '/dataset/OcularDisease/' --num_classes 3 --data_size '[1, 224, 224]' --source-domains 27 --intermediate-domains 2 --target-domains 4 --mode train --model-func resnet18 --feature-dim 512 --epochs 40 --iterations 100 --train_batch_size 28 --eval_batch_size 8 --test_epoch -1 --algorithm SDE --seed 0 --save_path './logs/' --record --mlp-depth 3 --mlp-width 512
'''

PowerSupply experiment
'''
python main.py --gpu_ids 0 --data_name PowerSupply --data_path '/dataset/PowerSupply/powersupply.arff' --num_classes 2 --data_size '[1, 2]' --source-domains 15 --intermediate-domains 5 --target-domains 10 --mode train --model-func Toy_Linear_FE --feature-dim 256 --epochs 50 --iterations 50 --train_batch_size 64 --eval_batch_size 50 --test_epoch -1 --algorithm SDE --seed 0 --save_path './logs/' --record --mlp-depth 3 --mlp-width 256
'''

Circle experiment
'''
python main.py --gpu_ids 0 --data_name ToyCircle --data_path '/dataset/Toy_Circle/half-circle.pkl' --num_classes 2 --data_size '[1, 2]' --source-domains 15 --intermediate-domains 5 --target-domains 10 --mode train --model-func Toy_Linear_FE --feature-dim 100 --epochs 20 --iterations 50 --train_batch_size 64 --eval_batch_size 50 --test_epoch -1 --algorithm SDE --seed 0 --save_path './logs/' --record --mlp-depth 2
'''

Sine experiment
'''
python -m pdb main.py --gpu_ids 0 --data_name ToySine --data_path '/dataset/Toy_Sine/sine_24.pkl' --num_classes 2 --data_size '[1, 2]' --source-domains 12 --intermediate-domains 4 --target-domains 8 --mode train --model-func Toy_Linear_FE --feature-dim 32 --epochs 10 --iterations 50 --train_batch_size 64 --eval_batch_size 50 --test_epoch -1 --algorithm SDE --seed 0 --save_path './logs/' --record --mlp-depth 2 --dropout 0.7
'''


## Acknowledgement
This code is implemented based on the [domainbed](https://github.com/facebookresearch/DomainBed) code. and [LSSAE] (https://github.com/WonderSeven/LSSAE) code.

## Citation
If you use this code for your research, please consider citing:
```
@inproceedings{
zeng2024latent,
title={Latent Trajectory Learning for Limited Timestamps under Distribution Shift over Time},
author={QIUHAO Zeng, Changjian Shui, Long-Kai Huang, Peng Liu, Xi Chen, Charles Ling and Boyu Wang},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=bTMMNT7IdW}
}
```