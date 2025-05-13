import jax.numpy as jnp
from flax import linen as nn

import torch

def pt_to_jax(path_state_dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state_dict = torch.load(path_state_dict, map_location=device)['model']

    weights = load_weights(state_dict)
    model = FrontnetJax()

    return model, weights

def load_weights(state_dict):
    conv_counter = 0
    bn_counter = 0
    fc_counter = 0
    weights = {'params': {}, 'batch_stats': {}}
    for key, tensor in zip(state_dict.keys(), state_dict.values()):
        if 'conv' in key:
            # [outC, inC, kH, kW] -> [kH, kW, inC, outC]
            conv_kernel = jnp.transpose(tensor.detach().cpu().numpy(), (2, 3, 1, 0))
            weights['params'][f'conv_{conv_counter}'] = {'kernel': conv_kernel}
            conv_counter += 1

        if 'bn' in key:
            bn = jnp.array(tensor.detach().cpu().numpy())

            if 'weight' in key:
                weights['params'].update({f'bn_{bn_counter}': {'scale': bn}})
            if 'bias' in key:
                weights['params'][f'bn_{bn_counter}'].update({'bias': bn})
            if 'mean' in key:
                weights['batch_stats'].update({f'bn_{bn_counter}': {'mean': bn}})
            if 'var' in key:
                weights['batch_stats'][f'bn_{bn_counter}'].update({'var': bn})
                bn_counter += 1
        
        if 'fc' in key:
            if 'weight' in key:
                # [outC, inC] -> [inC, outC]
                fc_weight = jnp.transpose(tensor.detach().cpu().numpy(), (1, 0))
                weights['params'][f'fc_{fc_counter}'] = {'kernel': fc_weight}
            if 'bias' in key:
                fc_bias = jnp.array(tensor.detach().cpu().numpy())
                weights['params'][f'fc_{fc_counter}'].update({'bias': fc_bias})
                fc_counter += 1

    return weights

class FrontnetJax(nn.Module):
  def setup(self):
    self.conv_0 = nn.Conv(features=32, kernel_size=(5,5), strides=(2,2), 
                          padding=((2,2), (2,2)), use_bias=False, name='conv_0')
    self.bn_0 = nn.BatchNorm(momentum=0.9, use_running_average=True, name='bn_0')
    # relu
    # maxpool


    # conv block 1
    self.conv_1 = nn.Conv(features=32, kernel_size=(3, 3), strides=(2,2), 
                          padding=((1,1), (1,1)), use_bias=False, name='conv_1')
    self.bn_1 = nn.BatchNorm(momentum=0.9, use_running_average=True, name='bn_1')
    # relu
    self.conv_2 = nn.Conv(32, kernel_size=(3, 3), strides=(1,1), 
                          padding=((1,1), (1,1)), use_bias=False, name='conv_2')
    self.bn_2 = nn.BatchNorm(momentum=0.9, use_running_average=True, name='bn_2')
    # relu

    # conv block 2
    self.conv_3 = nn.Conv(features=32*2, kernel_size=(3,3), strides=(2,2), 
                          padding=((1,1), (1,1)), use_bias=False, name='conv_3')
    self.bn_3 = nn.BatchNorm(momentum=0.9, use_running_average=True, name='bn_3')
    # relu
    self.conv_4 = nn.Conv(32*2, kernel_size=(3,3), strides=(1,1), 
                          padding=((1,1), (1,1)), use_bias=False, name='conv_4')
    self.bn_4 = nn.BatchNorm(momentum=0.9, use_running_average=True, name='bn_4')
    # relu

    # conv block 3
    self.conv_5 = nn.Conv(features=32*4, kernel_size=(3,3), strides=(2,2), 
                          padding=((1,1), (1,1)), use_bias=False, name='conv_5')
    self.bn_5 = nn.BatchNorm(momentum=0.9, use_running_average=True, name='bn_5')
    # relu
    self.conv_6 = nn.Conv(32*4, kernel_size=(3,3), strides=(1,1), 
                          padding=((1,1), (1,1)), use_bias=False, name='conv_6')
    self.bn_6 = nn.BatchNorm(momentum=0.9, use_running_average=True, name='bn_6')
    # relu

    self.fc = nn.Dense(features=4, name='fc_0')

  @nn.compact
  def __call__(self, x):
    conv_0 = self.conv_0(x)
    bn_0 = self.bn_0(conv_0)
    relu_0 = nn.relu(bn_0)
    max_pool = nn.max_pool(relu_0, window_shape=(2,2), strides=(2,2), padding=((0,0), (0, 0)))
    
    # conv block 1
    conv_1 = self.conv_1(max_pool)
    bn_1 = self.bn_1(conv_1)
    relu_1 = nn.relu(bn_1)
    conv_2 = self.conv_2(relu_1)
    bn_2 = self.bn_2(conv_2)
    relu_2 = nn.relu(bn_2)

    # conv block 2
    conv_3 = self.conv_3(relu_2)
    bn_3 = self.bn_3(conv_3)
    relu_3 = nn.relu(bn_3)
    conv_4 = self.conv_4(relu_3)
    bn_4 = self.bn_4(conv_4)
    relu_4 = nn.relu(bn_4)

    # conv block 3
    conv_5 = self.conv_5(relu_4)
    bn_5 = self.bn_5(conv_5)
    relu_5 = nn.relu(bn_5)
    conv_6 = self.conv_6(relu_5)
    bn_6 = self.bn_6(conv_6)
    relu_6 = nn.relu(bn_6)

    # [N, H, W, C] -> [N, C, H, W]
    x = jnp.transpose(relu_6, (0, 3, 1, 2))
    x = jnp.reshape(x, (x.shape[0], -1))
    out = self.fc(x)

    return out
  