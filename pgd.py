import sys
sys.path.append('.')
import torch
import torch.nn as nn
import torch.optim as optim

from projected_sinkhorn import conjugate_sinkhorn, projected_sinkhorn
from projected_sinkhorn import wasserstein_cost

def attack(X,y, net, epsilon=0.01, epsilon_iters=10, epsilon_factor=1.1, 
           p=2, kernel_size=5, maxiters=400, 
           alpha=0.1, xmin=0, xmax=1, normalize=lambda x: x, verbose=0, 
           regularization=1000, sinkhorn_maxiters=400, 
           ball='wasserstein', norm='l2'): 
    batch_size = X.size(0)
    epsilon = X.new_ones(batch_size)*epsilon
    C = wasserstein_cost(X, p=p, kernel_size=kernel_size)
    normalization = X.view(batch_size,-1).sum(-1).view(batch_size,1,1,1)
    X_ = X.clone()

    X_best = X.clone()
    err_best = err = net(normalize(X)).max(1)[1] != y
    epsilon_best = epsilon.clone()

    t = 0
    while True: 
        X_.requires_grad = True
        opt = optim.SGD([X_], lr=0.1)
        loss = nn.CrossEntropyLoss()(net(normalize(X_)),y)
        opt.zero_grad()
        loss.backward()

        with torch.no_grad(): 
            # take a step
            if norm == 'linfinity': 
                X_[~err] += alpha*torch.sign(X_.grad[~err])
            elif norm == 'l2': 
                X_[~err] += (alpha*X_.grad/(X_.grad.view(X.size(0),-1).norm(dim=1).view(X.size(0),1,1,1)))[~err]
            elif norm == 'wasserstein': 
                sd_normalization = X_.view(batch_size,-1).sum(-1).view(batch_size,1,1,1)
                X_[~err] = (conjugate_sinkhorn(X_.clone()/sd_normalization, 
                                               X_.grad, C, alpha, regularization, 
                                               verbose=verbose, maxiters=sinkhorn_maxiters
                                               )*sd_normalization)[~err]
            else: 
                raise ValueError("Unknown norm")

            # project onto ball
            if ball == 'wasserstein': 
                X_[~err] = (projected_sinkhorn(X.clone()/normalization, 
                                          X_.detach()/normalization, 
                                          C,
                                          epsilon,
                                          regularization, 
                                          verbose=verbose, 
                                          maxiters=sinkhorn_maxiters)*normalization)[~err]
            elif ball == 'linfinity': 
                X_ = torch.min(X_, X + epsilon.view(X.size(0), 1, 1,1))
                X_ = torch.max(X_, X - epsilon.view(X.size(0), 1, 1,1))
            else:
                raise ValueError("Unknown ball")
            X_ = torch.clamp(X_, min=xmin, max=xmax)
            
            err = (net(normalize(X_)).max(1)[1] != y)
            err_rate = err.sum().item()/batch_size
            if err_rate > err_best.sum().item()/batch_size:
                X_best = X_.clone() 
                err_best = err
                epsilon_best = epsilon.clone()

            if verbose and t % verbose == 0:
                print(t, loss.item(), epsilon.mean().item(), err_rate)
            
            t += 1
            if err_rate == 1 or t == maxiters: 
                break

            if t > 0 and t % epsilon_iters == 0: 
                epsilon[~err] *= epsilon_factor

    epsilon_best[~err] = float('inf')
    return X_best, err_best, epsilon_best
