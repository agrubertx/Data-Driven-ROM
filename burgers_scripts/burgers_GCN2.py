import time
import torch
import matplotlib.pyplot as plt

import sys, os
dir = os.path.dirname(__file__)
filename = os.path.abspath(os.path.join(dir, '..'))
sys.path.append(filename)

from src.utils import save_checkpoints, Assigner, train_network
from src.networks import NetGCN2


########## Training ##########

nu = torch.load('datasets/burgersData')
data = Assigner(nu)

model_name=f'testBurgersGCNN'   # for saving and loading

net = NetGCN2(256, 3, 3, 8, 1, 4, data.edge_index[0])
print(net)

train_network(net, model_name, data, 20, 0.0025, no_pde=True)


########## Testing ##########

data.coords = np.array([i/256 for i in range(256)])
device = 'cpu'

data.MtestNormed = torch.tensor(data.MtestNormed).float()
data.StestNormed = torch.tensor(data.StestNormed).float()
data.Stest = torch.tensor(data.Stest).float()
M = data.MtestNormed
S = data.StestNormed.unsqueeze(-1)

net.eval()
graphconv = torch.load(f'checkpoints/{model_name}/best_model.pth.tar')
net.load_state_dict(graphconv['state_dict'])

gcn_pred = net.decode(net.predict(M))
gcn_recon = data.Smin + data.Smaxminusmin * gcn_pred[:,:,0].detach().numpy()

gcn_recon = torch.tensor(gcn_recon)
gcnL1, gcnL2 = compute_errors(gcn_recon, data.Stest)

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
    ax.plot(data.coords, gcn_recon[pts[i]], linewidth=4, linestyle = 'solid', color='#018571', label='GCN')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), prop={'size': 28})
plt.tight_layout()
plt.show()
