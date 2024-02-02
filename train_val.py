import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models
import torch.distributed as dist
from torch.multiprocessing import Process
import os
from tensorboardX import SummaryWriter
import argparse
import torch.nn as nn
import math
import time
import optimizer
# import resnet
import copy
import lion_pytorch
from models import vgg16_bn
from models import resnet18, resnet20
from imagenetv2_pytorch import ImageNetV2Dataset, ImageNetValDataset
import progressbar
import wandb
#from imagenetv2_pytorch import ImageNetV2Dataset
################################## define norm
def compute_norm(parameter):
    total_norm = 0
    if isinstance(parameter, torch.Tensor):
        paramter=[parameter]

    for p in parameter:
        param_norm=p.data.norm(2)
        total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** (1/2)
    return total_norm
################################## define norm


def change_lr(optim,decay):
    for group in optim.param_groups:
        group['lr']*=decay

def train(args):
    
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.507,0.487,0.441], std = [0.267,0.256,0.276])
        ])
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                std=[0.267, 0.256, 0.276])])
    elif args.dataset == 'imagenet':
        transform_train = transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                            ])
        transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                            ])


    device = torch.device("cuda:%s"%(args.gpu[0]))

    print(args.max_iter)
    args.batch_size = args.batch_size
    if not os.path.isdir("%s"%(args.log_dir)):
        os.mkdir("%s"%(args.log_dir))

    task_name = f'lr-{args.lr}_theta-{args.theta}_zeta-{args.zeta}_alg-{args.Algorithm}/'
    writer = SummaryWriter(log_dir=args.log_dir + task_name)
    
    np.random.seed(args.rand_seed )
    torch.manual_seed(args.rand_seed)
    torch.cuda.manual_seed(args.rand_seed)
        
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='/ssd/datasets/cifar/', train=True,
                                            download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                        shuffle=True, num_workers=min(4,args.batch_size))
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='/ssd/datasets/cifar/', train=True,
                                            download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=min(4,args.batch_size))
    elif args.dataset == 'imagenet':
        trainset = torchvision.datasets.ImageFolder(root='/root/workspace/A_data/imagenet/train', 
                                                    transform=transform_train) 
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, 
                                                  shuffle=True, num_workers=min(8,args.batch_size))

    criterion = nn.CrossEntropyLoss()
    if args.dataset == 'cifar10':
        model = vgg16_bn(num_classes=10)
    elif args.dataset == 'cifar100':
        model = resnet20(num_classes=100)
    elif args.dataset == 'imagenet':
        model = resnet18(num_classes=1000)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    print(model)

    if (args.Algorithm == "ASIGNSGD"):
        optim = optimizer.ASIGNSGD(model.parameters(), lr = args.lr, theta = args.theta, weight_decay = args.weight_decay, zeta = args.zeta)
    elif (args.Algorithm == "SGDM"):
        optim = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = args.theta, weight_decay = args.weight_decay)
    elif (args.Algorithm == "ADAMW"):
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.theta, args.zeta), weight_decay = args.weight_decay)
    elif (args.Algorithm == "ASIGNSGDW"):
        optim = optimizer.ASIGNSGDW(model.parameters(), lr = args.lr, theta = args.theta, weight_decay = args.weight_decay, zeta = args.zeta)
    elif(args.Algorithm == "LION"):
        optim = lion_pytorch.Lion(model.parameters(), lr = args.lr, betas = (args.theta, args.zeta), weight_decay = args.weight_decay)
    else:
        print("NOT Implement!")
        return 0
    
    if args.dataset == 'cifar10':
        testset = torchvision.datasets.CIFAR10(root='/ssd/datasets/cifar/',train=False,download = True, transform = transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size = 128, shuffle = False, num_workers = 8)
    elif args.dataset == 'cifar100':
        testset = torchvision.datasets.CIFAR100(root='/ssd/datasets/cifar/',train=False,download = True, transform = transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size = args.batch_size, shuffle = False, num_workers = 8)
    elif args.dataset == 'imagenet':
        testset = torchvision.datasets.ImageFolder(root='/root/workspace/A_data/imagenet/val', 
                                                    transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size = args.batch_size,
                                                 shuffle=True, num_workers=min(8,args.batch_size))
    
    print('Data loaded and training start ...')
    step = 0
    batch_num = len(trainloader)
    for epoch in range(args.epoch):
        if (step>(args.max_iter)):
             break
        batch_count = 0
        train_loss_sum = 0
        train_ac = 0
        ac_total = 0
        for i, data in enumerate(trainloader):
            x,y = data
            step = step + 1
            batch_count += 1
            if (step%(args.decay_time)==0):
                 change_lr(optim, args.decay_lr)
            if (step>(args.max_iter)):
                 break
            y = y.to(device)
            x = x.to(device)
            # optimize
            model.zero_grad()
            if (args.Algorithm == "ASIGNSGD") or (args.Algorithm == "ASIGNSGDW"):
                optim.prev_step()
            y1 = model(x)
            loss = criterion(y1,y)
            y2 = torch.argmax(y1, dim = 1)
            loss.backward()
            optim.step()
            y2 = torch.argmax(y1, dim = 1)
            train_ac += torch.sum(y2==y).item()
            ac_total += torch.sum(y2==y2).item()
            train_loss_sum += loss
            # print("step:", step, "loss:", loss)
            print('Epoch: {} | Batch: {}/{} | Loss: {}'.format(epoch, batch_count, batch_num, loss.item()))
            # writer.add_scalar('loss', loss.data.item(), step)
            wandb.log({
                'Step': step,
                'Batch Train Loss': loss.item()
            })
            
            if ((step)%(len(trainloader))==0):
                # test_model = copy.deepcopy(model)
                # test_model.eval()
                model.eval()
                total = 0
                ac = 0
                total_loss = 0
                total_step = 0
                for j,data1 in enumerate(testloader):
                    tx,ty = data1
                    total_step += 1 
                    ty = ty.to(device)
                    tx = tx.to(device)
                    # ty1 = test_model(tx)
                    ty1 = model(tx)
                    test_loss = criterion(ty1,ty)
                    ty2 = torch.argmax(ty1,dim = 1)
                    ac = ac + torch.sum(ty2==ty).item()
                    total = total + torch.sum(ty2==ty2).item()
                    total_loss = total_loss + test_loss.data.item()
                    print('Epoch: {} | Batch: {}/{} | Batch Test Loss: {}'.format(epoch, i+1, len(testloader), loss.item()))
                wandb.log({
                    'Test Acc': ac/total,
                    'Test Loss': total_loss/total_step
                })
                print("step:",step," test loss:",total_loss/total_step," acc:",ac/total)
                model.train()

        train_loss_epoch = train_loss_sum / batch_count
        train_ac_epoch = train_ac / ac_total

########################################################### inner product
        smooth_count=0
        param_smooth = [param.data for param in model.parameters()]
        grad_smooth = [torch.zeros_like(w) for w in param_smooth]
        print('Computing the Smoothness ... ...')
        total = len(trainloader) 
        bar = progressbar.ProgressBar(maxval=total+1, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        recs = {}
        bar_counter = 0
        for i, data in enumerate(trainloader):
            x,y = data
            model.zero_grad()
            y = y.to(device)
            x = x.to(device)
            y1 = model(x)
            loss = criterion(y1,y)
            loss.backward()
            grad_iter = [param.grad.data for param in model.parameters()]
            for j in range(len(grad_iter)):
                grad_smooth[j] += grad_iter[j]
            smooth_count+=1
            # bar update
            recs[bar_counter] = bar_counter*bar_counter
            bar_counter += 1
            bar.update(bar_counter) 
            if bar_counter==total:
                break
        num_smooth=0
        for i in range(len(grad_smooth)):
            num_smooth += torch.dot(param_smooth[i].view(-1), grad_smooth[i].view(-1))
        dnom=compute_norm(param_smooth) * compute_norm(grad_smooth)
        smoothness = num_smooth.item() / dnom
        bar.finish()
        print("epoch:", epoch, "smooth_count:", smooth_count, "smoothness:", np.abs(smoothness))
        wandb.log({
                    'Epoch': epoch,
                    'Smoothness': np.abs(smoothness), 
                    'Train Loss': train_loss_epoch,
                    'Train Acc': train_ac_epoch
                })
        

###########################################################


    torch.save(model.state_dict(),args.filename)
    writer.close()
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type = int, default = 1)
    parser.add_argument('--gpu', type = str, default = "0")
    parser.add_argument("--filename", type = str, default = "model.pth")
    parser.add_argument("--log_dir", type = str, default = "logs/")
    parser.add_argument('--dataset', type = str, default = "imagenet")
    
    parser.add_argument("--lr",type = float, default = 1e-1)
    parser.add_argument("--theta",type = float, default =  0.9)
    parser.add_argument("--zeta", type = float, default = 0.2)
    
    parser.add_argument("--Algorithm", type = str, default = "SGDM")
    
    
    parser.add_argument("--epoch", type = int, default = 150)
    parser.add_argument("--batch_size", type = int, default = 128)
    parser.add_argument("--decay_time",type = int, default = 11730)
    parser.add_argument("--weight_decay",type = float, default = 5e-4)
    parser.add_argument("--decay_lr", type = float, default = 0.2) #congliang chen
    parser.add_argument("--rand_seed", type = int, default = 125025)
    parser.add_argument("--max_iter",type = int, default = 99999999)
    return parser.parse_args()

if __name__ == "__main__":

    processes = []
    args =  parse()
    print(args)
    kwargs = {
                'entity': '[YOUR ENTITY]', 
                'project': '[PROJECT NAME]',
                'mode': 'disabled',
                'name': 'algo: {}| lr: {}| theta: {}| zeta: {} | wd: {}| dlr: {}'.format(
                    args.Algorithm,
                    args.lr,
                    args.theta,
                    args.zeta,
                    args.weight_decay,
                    args.decay_lr
                ),
                'config': args,
                'settings': wandb.Settings(_disable_stats=True), 'reinit': True
             }
    wandb.init(**kwargs)
    wandb.save('*.txt')
    args.gpu = args.gpu.split(',')
    train(args)
