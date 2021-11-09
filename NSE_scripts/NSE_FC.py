import time
import numpy as np
import torch
import matplotlib.pyplot as plt

import sys, os
dir = os.path.dirname(__file__)
filename = os.path.abspath(os.path.join(dir, '..'))
sys.path.append(filename)

from src.networks import NetFCNN
from src.utils import save_checkpoints, Assigner, train_network

nu = torch.load('datasets/NSEData')
data = Assigner(nu)

model_name=f'testNSEFCNN'   # for saving and loading

net = NetFCNN(2, 10, 2, 4, [20208, 2021, 202])
print(net)

train_network(net, model_name, data, 32, 0.0005, stopNum=50, no_pde=False)


########## Testing ##########

# Load model
net.eval()
best_model_PDE = torch.load(f'checkpoints/{model_name}/best_model.pth.tar')
net.load_state_dict(best_model_PDE['state_dict'])

data.Sn = torch.tensor(data.Sn)
x = net(data.xTrain)

recon = data.Smin + data.Smaxminusmin * x.detach().numpy()

exactSpeed = torch.sqrt(torch.sum(data.snapshots*data.snapshots, -1))
reconSpeed = np.sqrt(np.sum(recon*recon, 2))

fig, axs = plt.subplots(3, sharex=True, sharey=True)
fig.suptitle('FCNN: Exact, Reconstructed, Error')
axs[0].scatter(data.coords[:,0], coords[:,1],
                c=exactSpeed[-4,-100].detach().numpy(), s=1)
axs[1].scatter(data.coords[:,0], coords[:,1],
                c=reconSpeed[-100], s=1)
axs[2].scatter(data.coords[:,0], coords[:,1],
                c=reconSpeed[-100] - exactSpeed[-4,-100].detach().numpy(), s=1)

recon = torch.tensor(recon)

# Compute errors
relL1 = torch.nn.L1Loss()(recon, data.snapshots[-4]) / torch.nn.L1Loss()(
            data.snapshots[-4], torch.zeros_like(data.snapshots[-4])) * 100
relL2 = torch.sqrt(torch.nn.MSELoss()(data.snapshots[-4], recon) /
            torch.nn.MSELoss()(data.snapshots[-4],
                torch.zeros_like(data.snapshots[-4]))) * 100

print(f'The relative L-1 Error is {relL1.item():.5f}%')
print(f'The relative L-2 Error is {relL2.item():.5f}%')
