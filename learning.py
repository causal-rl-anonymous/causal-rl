import torch
import datetime
import numpy as np
from utils import Dataset


def print_log(str, logfile=None):
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)


def fit_model(m, train_data, valid_data, loss_type='nll', with_done=False,
              n_epochs=200, epoch_size=100, batch_size=16,
              lr=1e-2, patience=10, log=False, logfile=None, min_int_ratio=0.0):

    # infer the device from the model
    device = next(m.parameters()).device

    if log:
        print_log(f"loss_type: {loss_type}", logfile)
        print_log(f"with_done: {with_done}", logfile)
        print_log(f"n_epochs: {n_epochs}", logfile)
        print_log(f"epoch_size: {epoch_size}", logfile)
        print_log(f"batch_size: {batch_size}", logfile)
        print_log(f"lr: {lr}", logfile)
        print_log(f"patience: {patience}", logfile)
        print_log(f"device: {device}", logfile)
        print_log(f"min_int_ratio: {min_int_ratio}", logfile)

    def compute_weights(data):
        nint = np.sum([regime == 1 for regime, _ in data])
        nobs = len(data) - nint
        int_ratio = nint / (nint + nobs)

        if int_ratio >= min_int_ratio:
            weights = [1] * len(data)
        else:
            weights = [(1 - min_int_ratio) / nobs, min_int_ratio / nint]  # obs, int
            weights = [weights[int(regime)] for regime, _ in data]

        return weights

    train_weights = compute_weights(train_data)
    valid_weights = compute_weights(valid_data)

    # Build training and validation data
    train_dataset = Dataset(train_data)
    valid_dataset = Dataset(list(zip(valid_data, valid_weights)))  # to reweight the loss

    sampler = torch.utils.data.WeightedRandomSampler(train_weights, replacement=True, num_samples=epoch_size*batch_size)

    # Initiate DataLoader for training and validation
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=sampler, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

    # Adam Optimizer with learning rate lr
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)

    # Scheduler. Reduce learning rate on plateau.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, verbose=log)

    # Early stopping
    best_valid_loss = float("Inf")
    best_parameters = m.state_dict().copy()
    best_epoch = -1

    # Start training loop
    for epoch in range(n_epochs + 1):

        # Set initial training loss as +inf 
        if epoch == 0:
            train_loss = float("Inf")

        else:
            train_loss = 0
            train_nsamples = 0

            for batch in train_loader:
                regime, episode = batch
                regime = regime.to(device)
                episode = [tensor.to(device) for tensor in episode] 

                batch_size = regime.shape[0]

                if loss_type == 'em':
                    loss = m.loss_em(regime, episode, with_done=with_done).mean()
                elif loss_type == 'nll':
                    loss = m.loss_nll(regime, episode, with_done=with_done).mean()
                elif loss_type == 'elbo':
                    raise NotImplementedError()
                else:
                    raise NotImplementedError()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_size
                train_nsamples += batch_size

            train_loss /= train_nsamples

        # validation
        valid_loss = 0
        valid_nsamples = 0

        for batch in valid_loader:
            (regime, episode), weight = batch
            regime = regime.to(device)
            episode = [tensor.to(device) for tensor in episode] 
            weight = weight.to(device)

            batch_size = regime.shape[0]
            
            with torch.no_grad():
                
                loss = m.loss_nll(regime, episode, with_done=with_done)
                loss = (loss * weight).sum()  # re-weighting the loss here

            valid_loss += loss.item()
            valid_nsamples += weight.sum().item()

        valid_loss /= valid_nsamples

        if log:
            print_log(f"epoch {epoch:04d} / {n_epochs:04d} train loss={train_loss:0.3f} valid loss={valid_loss:0.3f}", logfile)
#             q_s = torch.nn.functional.softmax(m.params_s.detach(), dim=-1)
#             print_log(f"  q_s: {((q_s.cpu().numpy() * 100) // 1) / 100}", logfile)

        # check for best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_parameters = m.state_dict().copy()
            best_epoch = epoch

        # check for early stopping
        if epoch > best_epoch + 2*patience:
            if log:
                print_log(f"{epoch-best_epoch} epochs without improvement, stopping.", logfile)
            break

        scheduler.step(valid_loss)

    # restore best model
    m.load_state_dict(best_parameters)



def eval_model(m, data, batch_size=32, with_done=False):

    """ Evaluate model m on data using Negative Log-Likehood on episodes """

    # infer the device from the model
    device = next(m.parameters()).device

    # Initialize NLL
    nll = 0

    # Load data as Dataset and create torch Dataloader 
    dataset = Dataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # Iterate on dataloader 
    for batch in dataloader:
        regime, episode = batch
        # Switch device if needed
        regime = regime.to(device)
        episode = [tensor.to(device) for tensor in episode] 

        # Get no grad NLL on batch
        with torch.no_grad():
            nll += m.loss_nll(regime, episode, with_done=with_done).sum().item()

    # Get mean NLL on data
    nll /= len(data)

    return nll
