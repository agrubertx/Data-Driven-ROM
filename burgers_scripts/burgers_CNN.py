import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

import sys, os
dir = os.path.dirname(__file__)
filename = os.path.abspath(os.path.join(dir, '..'))
sys.path.append(filename)

from src.networks import NetCNN_Burgers
from src.utils import save_checkpoints, Assigner, train_network


########## Training ##########

nu = torch.load('datasets/burgersData')
data = Assigner(nu)

model_name=f'testBurgersCNN'

net = NetCNN_Burgers(32,4)
print(net)

train_network(net, model_name, data, 20, 0.001, no_pde=False)


########## Testing ##########

data.coords = np.array([i/256 for i in range(256)])
device = 'cpu'

data.MtestNormed = torch.tensor(data.MtestNormed).float()
data.StestNormed = torch.tensor(data.StestNormed).float()
data.Stest = torch.tensor(data.Stest).float()
M = data.MtestNormed
S = data.StestNormed.unsqueeze(-1)

conv = torch.load(f'checkpoints/{model_name}/best_model.pth.tar')
net.load_state_dict(conv['state_dict'])

cnn_pred = net.decode(net.predict(M))
cnn_recon = data.Smin + data.Smaxminusmin * cnn_pred[:,:,0].detach().numpy()
fcnn_pred = net_fcnn.decode(net_fcnn.predict(M))

cnn_recon = torch.tensor(cnn_recon)
cnnL1, cnnL2 = compute_errors(cnn_recon, data.Stest)

print('Errors in original vs reconstructed solution:')
print(f'The relative L-1 Error is {cnnL1.item():.5f}% for CNN, {gcnL1.item():.5f}% for GCN, and {fcnnL1.item():.5f}% for FCNN')
print(f'The relative L-2 Error is {cnnL2.item():.5f}% for CNN, {gcnL2.item():.5f}% for GCN, and {fcnnL2.item():.5f}% for FCNN')

# pts = [525, 551, 576, 601]
pts = [525, 560, 595]

fig = plt.figure(1, figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.xaxis.label.set_size(24)
ax.yaxis.label.set_size(24)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
ax.set_xlabel('Position $x$', labelpad=15)
ax.set_ylabel('Solution $w$', labelpad=15)
for i in range(3):
    ax.plot(data.coords, data.Stest[pts[i]], linewidth = 10, linestyle = 'solid', color='#A6611A', label='Exact')
    ax.plot(data.coords, cnn_recon[pts[i]], linewidth=4, linestyle = 'solid', color='#80CDC1', label='CNN')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), prop={'size': 28})
plt.tight_layout()
plt.show()
