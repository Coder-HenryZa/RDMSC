import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.accs=0
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, accs,model,modelname,str):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.accs = accs
        elif score < self.best_score:
            self.counter += 1
            # print('EarlyStopping counter: {} out of {}'.format(self.counter,self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.accs = accs
            self.counter = 0

    def save_checkpoint(self, val_loss, model,modelname,str):

        torch.save(model.state_dict(),modelname+str+'.m')
        self.val_loss_min = val_loss