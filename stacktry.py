import numpy as np
import torch


# b=1
# w = h = 32
# grid = np.meshgrid(range(32), range(32))
# # print(grid)
#
# grid = np.stack(grid, axis=-1)
# grid[:, :, 0] = grid[:, :, 0]*2
# grid[:, :, 0] = grid[:, :, 0] / (w - 1) -1 #w축
# grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) -1 #h축
# grid = grid.transpose(2, 0, 1) # (32, 32, 2) -> (2, 32, 32)
# grid = np.tile(grid, (b, 1, 1, 1)) # (2, 32, 32) -> (128, 2, 32, 32)

input = torch.arange(4*4).view(1,1,4,4).float()
print(input)

d = torch.linspace(-1,1,8)
print(d)

x, y = torch.meshgrid((d,d))
grid = torch.stack((x,y), 2)
grid = grid.unsqueeze(0)

output = torch.nn.functional.grid_sample(input, grid)
print(output)
