# Investigating the Role of Weight Decay in Enhancing Nonconvex SGD

The repositroy contains the codes for the paper [Investigating the Role of Weight Decay in Enhancing Nonconvex SGD](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_Investigating_the_Role_of_Weight_Decay_in_Enhancing_Nonconvex_SGD_CVPR_2025_paper.pdf) published on CVPR 2025. 

## Requirements
```
torch
torchvision
tensorboardX
lion_pytorch
```
    
## Runing Scripts
```
python train_val.py --dataset [cifar10/cifar100/imagenet] \
                    --Algorithm [SGDM/ASIGNSGD] --lr [learning_rate] \
                    --weight_decay [weight decay] \
                    --theta [theta] --zeta [zeta] 
```
or runing in parallel on different GPUs
```
CUDA_DEVICES_VISIBLE=0,1,2,3 python train_val.py --dataset [cifar10/cifar100/imagenet] \
                                                 --Algorithm [SGDM/ASIGNSGD] \
                                                 --lr [learning_rate] --weight_decay [weight decay] \
                                                 --theta [theta] --zeta [zeta] 
```
