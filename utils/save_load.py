import os
import torch


def save_checkpoint(model, optimizer, step, ckpdir, f_name, loss):
    ''''
    Ckpt saver
        f_name - <str> the name phnof ckpt file (w/o prefix) to store, overwrite if existed
        score  - <float> The value of metric used to evaluate model
    '''
    ckpt_path = os.path.join(ckpdir, f_name)
    full_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.get_opt_state_dict(),
        "global_step": step,
        "loss": loss
    }
    # Additional modules to save
    # if self.amp:
    #    full_dict['amp'] = self.amp_lib.state_dict()

    torch.save(full_dict, ckpt_path)

def load_ckpt(model, optimizer, device, path, mode='train'):
    ''' Load ckpt if --load option is specified '''
    step = None
    # Load weights
    ckpt = torch.load(path, map_location=device if mode == 'train' else 'cpu')
    model.load_state_dict(ckpt['model'])
    if mode == 'train':
        step = ckpt['global_step']
        optimizer.load_opt_state_dict(ckpt['optimizer'])
    return model, optimizer, step