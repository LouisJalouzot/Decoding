import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence

from src.utils import console


def compute_mse_loss(X, Y, decoder):
    Y_preds = decoder(X)
    return F.mse_loss(Y_preds, Y)


def compute_symm_nce_loss(X, Y, decoder, temperature):
    Y_preds = decoder(X)
    Y_preds = F.normalize(Y_preds, dim=-1)
    Y = F.normalize(Y, dim=-1)
    logits = Y_preds @ Y.T / temperature
    target = torch.arange(logits.shape[0]).to(logits.device)
    loss = F.cross_entropy(logits, target)
    loss2 = F.cross_entropy(logits.T, target)
    return (loss + loss2) / 2


def mixco_sample_augmentation(X, beta=0.15, s_thresh=0.5):
    """Augment samples with MixCo augmentation.

    Parameters
    ----------
    samples : torch.Tensor
        Samples to augment.
    beta : float, optional
        Beta parameter for the Beta distribution, by default 0.15
    s_thresh : float, optional
        Proportion of samples which should be affected by MixCo, by default 0.5

    Returns
    -------
    samples : torch.Tensor
        Augmented samples.
    perm : torch.Tensor
        Permutation of the samples.
    betas : torch.Tensor
        Betas for the MixCo augmentation.
    select : torch.Tensor
        Samples affected by MixCo augmentation
    """
    # Randomly select samples to augment
    if isinstance(X, PackedSequence):
        samples = X.data
    else:
        samples = X

    select = (torch.rand(samples.shape[0]) <= s_thresh).to(samples.device)

    # Randomly select samples used for augmentation
    perm = torch.randperm(samples.shape[0])
    samples_shuffle = samples[perm].to(samples.device, dtype=samples.dtype)

    # Sample MixCo coefficients from a Beta distribution
    betas = (
        torch.distributions.Beta(beta, beta)
        .sample([samples.shape[0]])
        .to(samples.device, dtype=samples.dtype)
    )
    betas[~select] = 1
    betas_shape = [-1] + [1] * (len(samples.shape) - 1)

    # Augment samples
    samples[select] = samples[select] * betas[select].reshape(
        *betas_shape
    ) + samples_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)

    if isinstance(X, PackedSequence):
        X = PackedSequence(samples, X.batch_sizes, X.sorted_indices, X.unsorted_indices)
    else:
        X = samples

    return X, perm, betas


def compute_mixco_symm_nce_loss(X, Y, decoder, temperature):
    X_mixco, perm, betas = mixco_sample_augmentation(X)
    Y_preds = decoder(X_mixco)
    Y_preds = F.normalize(Y_preds, dim=-1)
    Y = F.normalize(Y, dim=-1)
    logits = Y_preds @ Y.T / temperature
    probs = torch.diag(betas)
    probs[torch.arange(Y_preds.shape[0]).to(Y_preds.device), perm] = 1 - betas
    loss = -(logits.log_softmax(-1) * probs).sum(-1).mean()
    loss2 = -(logits.T.log_softmax(-1) * probs.T).sum(-1).mean()
    return (loss + loss2) / 2
