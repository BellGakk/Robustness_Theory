import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn

class FGSMAttack(object):
    def __init__(self, model=None, epsilon=None, loss_fn=None):

        self.model = model
        self.epsilon = epsilon
        self.loss_fn = loss_fn

    def perturb(self, original_imgs, labels):
        
        adv_imgs = original_imgs.clone()
        original_imgs.requires_grad = True
        scores = self.model(original_imgs)
        loss = self.loss_fn(scores, labels)
        loss.backward()
        grad_sign = original_imgs.grad.data.sign()

        adv_imgs += self.epsilon * grad_sign
        adv_imgs = torch.clip(adv_imgs, -1, 1)

        return adv_imgs

'''
class LinfPGDAttack(object):
    def __init__(self, model=None, loss_fn=None, epsilon=0.3, iter=40, alpha=0.01, 
        random_start=True):

        self.model = model
        self.epsilon = epsilon
        self.iter = iter
        self.alpha = alpha
        self.rand = random_start
        self.loss_fn = loss_fn

    def perturb(self, original_imgs, labels):

        if self.rand:
            adv_imgs = original_imgs + np.random.uniform(-self.epsilon, self.epsilon,
                X_nat.shape).astype('float32')
        else:
            X = np.copy(

        for i in range(self.k):
            X_var = to_var(torch.from_numpy(X), requires_grad=True)
            y_var = to_var(torch.LongTensor(y))

            scores = self.model(X_var)
            loss = self.loss_fn(scores, y_var)
            loss.backward()
            grad = X_var.grad.data.cpu().numpy()

            X += self.a * np.sign(grad)

            X = np.clip(X, X_nat - self.epsilon, X_nat + self.epsilon)
            X = np.clip(X, 0, 1) # ensure valid pixel range

        return X
'''