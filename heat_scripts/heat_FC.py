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


########## Training ##########

nu = torch.load('datasets/heatData')
data = Assigner(nu)

model_name=f'testHeatFC'   # for saving and loading

net = NetFCNN(1, 10, 3, 4, [2145, 215])
print(net)

train_network(net, model_name, data, 32, 0.0005, stopNum=100, no_pde=False)


########## Testing ##########

data.UTestNormed = torch.tensor(data.UTestNormed).float()
data.UTest = torch.tensor(data.UTest).float()
data.PTestNormed = torch.tensor(data.PTestNormed).float()
grid_x, grid_y = np.mgrid[0:1:100j, 0:2:200j]

pad = torch.zeros(1600,1951,1)
Ucnn = torch.cat((data.UTestNormed.unsqueeze(-1), pad), 1)

net.eval()
fullcon = torch.load(f'checkpoints/{model_name}/best_model.pth.tar')
net.load_state_dict(fullcon['state_dict'])

fcnn_pred = net.decode(net.predict(data.PTestNormed))
fcnn_recon = data.Umin + data.Umaxminusmin * fcnn_pred[:,:,0].detach().numpy()

fcnn_recon = torch.tensor(fcnn_recon)
fcnnL1, fcnnL2 = compute_errors(fcnn_recon, data.UTest)

print('Errors in original vs reconstructed solution:')
print(f'The relative L-1 Error is {fcnnL1.item():.5f}% for FCNN')
print(f'The relative L-2 Error is {fcnnL2.item():.5f}% for FCNN')

list = [100, 1000, 1500]
words = ['Exact', 'FCNN']
cnn = gcnn = fcnn = exact = [0,0,0]
fig, ax = plt.subplots(3, 2, subplot_kw={"projection": "3d"}, figsize=(10,7))
# plt.suptitle('Predicted Samples', fontsize=16)
for i in range(3):
    exact[i] = griddata(data.coords, data.UTest[list[i]], (grid_x, grid_y), method='cubic')
    fcnn[i] = griddata(data.coords, fcnn_recon[list[i]], (grid_x, grid_y), method='cubic')
for i in range(3):
    for j in range(2):
        ax[i][j].axes.xaxis.set_ticklabels([])
        ax[i][j].axes.yaxis.set_ticklabels([])
        ax[i][j].axes.zaxis.set_ticklabels([])
        ax[0][j].set_title(f'{words[j]}')
        ax[i][j].view_init(30, 125)
for i in range(3):
    ax[i][0].plot_surface(grid_x, grid_y, exact[i], cmap='viridis')
    ax[i][1].plot_surface(grid_x, grid_y, fcnn[i], cmap='viridis')
# plt.tight_layout()
plt.show()
