import math
import torch
import copy
import time 
from torch.optim.optimizer import Optimizer


class ASIGNSGD(Optimizer):
    def __init__(self,params,lr, weight_decay, theta, zeta):
        defaults = dict(lr = lr, weight_decay=weight_decay, theta=theta,zeta = zeta)
        super(ASIGNSGD,self).__init__(params,defaults)
    
    def prev_step(self):
        for group in self.param_groups:
           for p in group['params']:
             state = self.state[p]
             if (len(state)==0):
                    state['step'] = 0
                    state['x'] = p.data.clone()
                    state['update'] = torch.zeros_like(p.data)
                    state['m']= torch.zeros_like(p.data)
             p.data = state['x'] + group['zeta']*state['update']
             
             
    def step(self,closure=None):
       loss = None
       if closure is not None:
            loss = closure()
       for group in self.param_groups:
           for p in group['params']:
               if p.grad is None:
                  continue
                                
               grad = p.grad.data
               state = self.state[p]
               if group['weight_decay']!=0:
                   grad.add_(state['x'], alpha = group['weight_decay'])
               
               theta = group['theta']
               state['m'] = theta*state['m'] + (1-theta)*grad
               state['update'] = -group['lr']*torch.sign(state['m'])
               p.data = state['x'] + state['update']
               state['x'] = p.data.clone()
       return loss        
       
    def change_lr(self,decay):
        for group in self.param_groups:
               group['lr']*=decay




class ASIGNSGDW(Optimizer):
    def __init__(self,params,lr, weight_decay, theta, zeta):
        defaults = dict(lr = lr, weight_decay=weight_decay, theta=theta,zeta = zeta)
        super(ASIGNSGDW,self).__init__(params,defaults)
    
    def prev_step(self):
        for group in self.param_groups:
           for p in group['params']:
             state = self.state[p]
             if (len(state)==0):
                    state['step'] = 0
                    state['x'] = p.data.clone()
                    state['update'] = torch.zeros_like(p.data)
                    state['m']= torch.zeros_like(p.data)
             p.data = state['x'] + group['zeta']*state['update']
             
             
    def step(self,closure=None):
       loss = None
       if closure is not None:
            loss = closure()
       for group in self.param_groups:
           for p in group['params']:
               if p.grad is None:
                  continue
                                
               grad = p.grad.data
               state = self.state[p]
              
               
               theta = group['theta']
               state['m'] = theta*state['m'] + (1-theta)*grad
               state['update'] = -group['lr']*torch.sign(state['m'])
               if group['weight_decay']!=0:
                   state['update'] = state['update'] - group['lr']*group['weight_decay']*p.data 
               p.data = state['x'] + state['update']
               state['x'] = p.data.clone()
       return loss        
       
    def change_lr(self,decay):
        for group in self.param_groups:
               group['lr']*=decay


def compress(x):
    a = torch.abs(x)
    b = torch.max(a)
    a = a/b
    a = torch.round(a)
    return torch.sign(x)*b*a

class ASIGNSGDW_DIS(Optimizer):
    def __init__(self,params,lr, weight_decay, theta, zeta, N, K):
        defaults = dict(lr = lr, weight_decay=weight_decay, theta=theta,zeta = zeta,N = N, K = K)
        super(ASIGNSGDW_DIS,self).__init__(params,defaults)
        self.count = 0
        self.N = N
    def prev_step(self):
        for group in self.param_groups:
           for p in group['params']:
             state = self.state[p]
             if (len(state)==0):
                    state['step'] = 0
                    state['x'] = p.data.clone()
                    state['update'] = torch.zeros_like(p.data)
                    state['m']= torch.zeros_like(p.data)
                    state['grad'] = torch.zeros_like(p.data)
                    state['count'] = 0
             p.data = state['x'] + group['zeta']*state['update']
             state['grad'].zero_()
             
    """         
    def step(self,closure=None):
       loss = None
       self.count += 1
       if closure is not None:
            loss = closure()
       for group in self.param_groups:
           for p in group['params']:
               if p.grad is None:
                  continue
                                
               grad = p.grad.data
               state = self.state[p]
               v = torch.zeros_like(grad)
               for i in range(group['K']):
                  v = v + compress(grad-v)
               state['grad'] = state['grad']+v
       if (self.count == self.N):
          self.count = 0
          tmp = 0
          for group in self.param_groups:
           for p in group['params']:
               if p.grad is None:
                  continue
               state = self.state[p]
               grad = state['grad']/group['N']
               theta = group['theta']
               state['m'] = theta*state['m'] + (1-theta)*grad
               tmp = tmp + torch.norm(state['m'])**2
          tmp = torch.sqrt(tmp)
          for group in self.param_groups:
           for p in group['params']:
               if p.grad is None:
                  continue
               state =self.state[p]
               state['update'] = -group['lr']*state['m']/tmp
               if group['weight_decay']!=0:
                    state['update'] = state['update'] - group['lr']*group['weight_decay']*state['x'] 
               p.data = state['x'] + state['update']
               state['x'] = p.data.clone()
       return loss        
    """
    
    def step(self,closure=None):
       loss = None
       if closure is not None:
            loss = closure()
       for group in self.param_groups:
           for p in group['params']:
               if p.grad is None:
                  continue
                                
               grad = p.grad.data
               state = self.state[p]
               count = state['count']
               v = torch.zeros_like(grad)
               for i in range(group['K']):
                  v = v + compress(grad-v)
               state['grad'] = state['grad']+v
               count= count + 1
               if (count == self.N):
                    count = 0
                    grad = state['grad']/group['N']
                    theta = group['theta']
                    state['m'] = theta*state['m'] + (1-theta)*grad
                    tmp = torch.norm(state['m'])**2
                    tmp = torch.sqrt(tmp)
                    state['update'] = -group['lr']*state['m']/tmp
                    if group['weight_decay']!=0:
                        state['update'] = state['update'] - group['lr']*group['weight_decay']*state['x'] 
                    p.data = state['x'] + state['update']
                    state['x'] = p.data.clone()
               state['count'] = count
       return loss
    def change_lr(self,decay):
        for group in self.param_groups:
               group['lr']*=decay


class ASIGNSGD_DIS(Optimizer):
    def __init__(self,params,lr, weight_decay, theta, zeta, N, K):
        defaults = dict(lr = lr, weight_decay=weight_decay, theta=theta,zeta = zeta,N = N, K = K)
        super(ASIGNSGD_DIS,self).__init__(params,defaults)
        self.count = 0
        self.N = N
    def prev_step(self):
        for group in self.param_groups:
           for p in group['params']:
             state = self.state[p]
             if (len(state)==0):
                    state['step'] = 0
                    state['x'] = p.data.clone()
                    state['update'] = torch.zeros_like(p.data)
                    state['m']= torch.zeros_like(p.data)
                    state['grad'] = torch.zeros_like(p.data)
                    state['count'] = 0
             p.data = state['x'] + group['zeta']*state['update']
             state['grad'].zero_()
             
    """         
    def step(self,closure=None):
       loss = None
       self.count += 1
       if closure is not None:
            loss = closure()
       for group in self.param_groups:
           for p in group['params']:
               if p.grad is None:
                  continue
                                
               grad = p.grad.data
               state = self.state[p]
               v = torch.zeros_like(grad)
               for i in range(group['K']):
                  v = v + compress(grad-v)
               state['grad'] = state['grad']+v
       if (self.count == self.N):
          self.count = 0
          tmp = 0
          for group in self.param_groups:
           for p in group['params']:
               if p.grad is None:
                  continue
               state = self.state[p]
               grad = state['grad']/group['N']
               theta = group['theta']
               state['m'] = theta*state['m'] + (1-theta)*grad
               tmp = tmp + torch.norm(state['m'])**2
          tmp = torch.sqrt(tmp)
          for group in self.param_groups:
           for p in group['params']:
               if p.grad is None:
                  continue
               state =self.state[p]
               state['update'] = -group['lr']*state['m']/tmp
               if group['weight_decay']!=0:
                    state['update'] = state['update'] - group['lr']*group['weight_decay']*state['x'] 
               p.data = state['x'] + state['update']
               state['x'] = p.data.clone()
       return loss        
    """
    
    def step(self,closure=None):
       loss = None
       if closure is not None:
            loss = closure()
       for group in self.param_groups:
           for p in group['params']:
               if p.grad is None:
                  continue
               
               state = self.state[p]                 
               grad = p.grad.data + group['weight_decay']*state['x']
               #state = self.state[p]
               count = state['count']
               v = torch.zeros_like(grad)
               for i in range(group['K']):
                  v = v + compress(grad-v)
               state['grad'] = state['grad']+v
               count = count + 1
               if (count == self.N):
                    count = 0
                    grad = state['grad']/group['N'] #+ group['weight_decay']*state['x']
                    theta = group['theta']
                    state['m'] = theta*state['m'] + (1-theta)*grad
                    tmp = torch.norm(state['m'])**2
                    tmp = torch.sqrt(tmp)
                    state['update'] = -group['lr']*state['m']/tmp
                    p.data = state['x'] + state['update']
                    state['x'] = p.data.clone()
               state['count'] = count
       return loss
    def change_lr(self,decay):
        for group in self.param_groups:
               group['lr']*=decay


