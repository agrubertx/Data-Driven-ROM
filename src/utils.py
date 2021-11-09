import numpy as np
import torch
from torch.nn import MSELoss, L1Loss
import os, time, pickle
import shutil

# Loads dictionary into object
class Assigner(object):
    def __init__(self, dict):
        for key in dict:
            setattr(self, key, dict[key])


# Saves training checkpoints
def save_checkpoints(state, is_best=None,
                     base_dir='checkpoints',
                     save_dir=None):
    if save_dir:
        save_dir = os.path.join(base_dir, save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    checkpoint = os.path.join(save_dir, 'checkpoint.pth.tar')
    torch.save(state, checkpoint)
    if is_best:
        best_model = os.path.join(save_dir, 'best_model.pth.tar')
        shutil.copyfile(checkpoint, best_model)


def evaluate(network, x, p):
    '''
    Evaluate quantities necessary for computing loss.
    '''
    red = network.predict(p)
    z = network.encode(x)
    xTilde = network.decode(red)
    return z, red, xTilde


def train_network(network, network_name, data, batch_size, learning_rate,
                  stopNum=200, no_pde=False):
    '''
    Implements batch descent using the ADAM optimizer.  Iteration stops after
    stopNum consecutive epochs without progress.  Network is saved in the
    directory checkpoints/{network_name}.
    '''
    trainLoss_list = []
    validLoss_list = []
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    tt=time.time()
    network.train()
    print(f'no_pde is {no_pde}. ')

    with torch.no_grad():
        if no_pde:
            xTilde = network(data.xTrain)
            trainLoss = MSELoss()(xTilde, data.xTrain)
            xTilde = network(data.xValid)
            validLoss = MSELoss()(xTilde, data.xValid)
        else:
            z, red, xTilde = evaluate(network, data.xTrain, data.pTrain)
            trainLoss = MSELoss()(xTilde, data.xTrain) + MSELoss()(z, red)
            z, red, xTilde = evaluate(network, data.xValid, data.pValid)
            validLoss = MSELoss()(xTilde, data.xValid) + MSELoss()(z, red)

    trainLoss_list.append(float(trainLoss))
    validLoss_list.append(float(validLoss))
    # print(float(trainLoss), float(validLoss))

    print('Training Start...')
    count = 0
    best_loss = 1.
    for epoch in range(10000):
        total_trainloss = 0.
        permutation = torch.randperm(data.xTrain.size(0))

        if count == stopNum:
            with open(f'trainLoss_list_{network_name}.txt', 'w') as l:
                for item in trainLoss_list:
                    l.write("%s\n" % item)
            with open(f'validLoss_list_{network_name}.txt', 'w') as l:
                for item in validLoss_list:
                    l.write("%s\n" % item)
            break

        for i in range(0, data.xTrain.size(0), batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_x, batch_p = data.xTrain[indices], data.pTrain[indices]

            if no_pde:
                xTilde = network(batch_x)
                trainLoss = MSELoss()(xTilde, batch_x)
            else:
                z, red, xTilde = evaluate(network, batch_x, batch_p)
                trainLoss = MSELoss()(xTilde, batch_x) + MSELoss()(z, red)

            trainLoss.backward()
            optimizer.step()

            total_trainloss += float(trainLoss)

        if (epoch+1)%1 == 0:
            with torch.no_grad():
                network.eval()
                if no_pde:
                    xTilde = network(data.xValid)
                    validLoss = MSELoss()(xTilde, data.xValid)
                else:
                    z, red, xTilde = evaluate(network, data.xValid, data.pValid)
                    validLoss = MSELoss()(xTilde, data.xValid) + MSELoss()(z, red)

            print(f'#{epoch+1:5d}: validation_loss={validLoss.item():.3e},' +
                  f' training_loss={trainLoss.item():.3e},' +
                  f' time={time.time()-tt:.2f}s')

            trainLoss_list.append(trainLoss.item())
            validLoss_list.append(validLoss.item())

            is_best = validLoss.item() < best_loss
            state = {
                'epoch': epoch,
                'state_dict': network.state_dict(),
                'best_loss': best_loss
            }
            if is_best:
                best_loss = validLoss.item()
                print('new best!')
                count = 0
            else: count += 1
            save_checkpoints(state, is_best, save_dir=network_name)
            tt = time.time()
    print('Training Finished!')


def compute_errors(x, y):
    '''
    Computes relative L1/L2 errors.
    '''
    relL1 = L1Loss()(x,y) / L1Loss()(y, torch.zeros_like(y)) * 100
    relL2 = torch.sqrt(MSELoss()(x,y) / MSELoss()(y, torch.zeros_like(y))) * 100
    return relL1, relL2
