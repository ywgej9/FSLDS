import numpy as np
import torch
import torch.distributions as td
from scipy.stats import genpareto         


def trans_mat(p=0.05, D=2):
    """Create a transition matrix with a fixed probability."""
    mat = p * np.ones((D, D)) + np.eye(D)
    return mat / np.sum(mat[0, :])

def generate_2d_ar1_process(alpha, beta, n, d, seed, sigma=1):
    """Generate a 2-dimensional AR(1) process given coefficients and length of time series."""
    np.random.seed(seed)
    ts = np.zeros((n, d))
    for i in range(1, n):
        for j in range(d):
            ts[i, j] = alpha[j] + beta[j] * ts[i-1, j] + np.random.normal(scale=sigma)
    return ts


def create_mask(data_shape, mask_patterns):
    """
    Create a mask for the data tensor based on multiple masking patterns.

    Parameters:
    - data_shape: Tuple of (num_nodes, T), the shape of your data.
    - mask_patterns: List of dictionaries, each specifying a masking pattern with keys:
        - 'start_time': The time index to start masking from.
        - 'interval': The interval at which to apply the mask.
        - 'node_indices': The indices of the nodes to mask (list or slice).

    Returns:
    - mask: A boolean NumPy array of the same shape as the data, where True indicates a masked value.
    """

    T, num_nodes = data_shape
    mask = np.zeros((T, num_nodes), dtype=bool)

    for pattern in mask_patterns:
        start_time = pattern['start_time']
        interval = pattern['interval']
        node_indices = pattern['node_indices']

        # Generate time indices to mask starting from 'start_time' every 'interval' steps
        times_to_mask = np.arange(start_time, T, interval)

        # Apply the mask to the specified nodes at the specified times
        mask[times_to_mask, node_indices] = True

    return mask


def adjust_theta_learning_rate(optimizer, epoch, theta_initial_lr, theta_lr_after_threshold, epoch_threshold):
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'theta':
            if epoch < epoch_threshold:
                param_group['lr'] = theta_initial_lr
            else:
                param_group['lr'] = theta_lr_after_threshold
                
                
def compute_rmse_masked(true_data, predicted, mask):
    """
    true_data: (T, D) ground-truth observations
    predicted: (T, D) model predictions (e.g., mu_total)
    mask: (T, D) with 1 = observed and 0 = held-out.
    
    Returns the RMSE computed *only* over held-out (masked=0) entries.
    """
    # Convert mask=0 -> 1 for the missing positions (held-out),
    # so we can multiply by those positions to select them.
    missing_mask = mask  # shape (T,D)
    
    # Squared errors over all positions
    sq_error = (true_data - predicted)**2
    
    # Only keep the missing positions (where missing_mask=1)
    sq_error_missing = sq_error * missing_mask
    
    # Sum of squared errors over missing positions
    sum_sq_error = sq_error_missing.sum()
    # Count of missing positions
    count_missing = missing_mask.sum()
    
    if count_missing.item() == 0:
        # Edge case: if no missing data, return 0 or NaN
        return float('nan')
    
    # MSE => sqrt => RMSE
    rmse_missing = torch.sqrt(sum_sq_error / count_missing)
    return rmse_missing.item()

def fill_missing_times_with_previous(data, missing_times):
    """
    data: (T, D) tensor
    missing_times: a list of time steps where the entire row is missing 
                   (e.g., [10, 20, 30, ...])
    Returns a new tensor with each missing row t replaced by the row t-1.
    """

    data_imputed = data.clone()
    for t in missing_times:
        # Copy row t-1 into row t
        data_imputed[t, :] = data_imputed[t-1, :]

    return data_imputed


@torch.no_grad()
def posterior_predictive_check(
        model,                       # trained model (already eval-mode outside)
        data_tensor,                 # full data on *device*
        S: int = 2000,                 # number of importance samples
        device: torch.device = "cpu"
    ):
    """
    Returns
    -------
    log_weights : list[float]  (length S)
        Importance weights  log p(y, h, z, z0) – log q(h, z, z0 | y)
        averaged over time for each draw.
    """
    T, _ = data_tensor.shape
    log_ws = []

    for _ in range(S):
        # forward pass – rsample() produces one draw of (h, z, z0)
        h_logit, log_q_h, z, log_q_z, z0, log_q_z0, rho_logit = model(data_tensor)

        # ---------- log p(y | ⋯) ----------
        h_t       = torch.sigmoid(h_logit)
        mu_t      = h_t * torch.exp(z)        @ torch.abs(model.theta)
        mu0_t     = torch.exp(z0)           @ torch.abs(model.theta0)
        mu_total  = mu_t + mu0_t                                # (T, D)
        log_p_y   = td.Poisson(mu_total).log_prob(data_tensor)  # (T, D)
        log_p_y   = log_p_y.sum(dim=1).mean()                  # scalar

        # ---------- log p(z | z-1) ----------
        z_loss_dist  = td.MultivariateNormal(
            z[:-1] @ torch.diag(model.A), torch.diag(torch.abs(model.sigp[:, 0])))
        log_p_z      = z_loss_dist.log_prob(z[1:]).mean()

        # ---------- log p(h | h-1) ----------
        log_p_h = 0.
        temp = model.temp
        for k in range(model.K):
            dist_h  = td.relaxed_bernoulli.LogitRelaxedBernoulli(
                temp, logits=rho_logit[k][:-1])
            log_p_h += dist_h.log_prob(h_logit[1:, k].unsqueeze(-1)).mean()

        # ---------- log p(z0 | z0-1) ----------
        z0_loss_dist = td.Normal(model.A0 * z0[:-1], torch.abs(model.sigp0))
        log_p_z0     = z0_loss_dist.log_prob(z0[1:]).mean()

        # ---------- joint (expectation over time) ----------
        log_p = log_p_y + log_p_z + log_p_z0 + log_p_h

        # ---------- log q ----------
        log_q = log_q_h.sum(dim=1).mean() + log_q_z.mean() + log_q_z0.mean()

        log_ws.append((log_p - log_q).item())

    return log_ws

@torch.no_grad()
def psis_khat(log_w, tail_frac: float = 0.2):
    """
    Pareto-Smoothed Importance Sampling (PSIS) diagnostic.
    Returns the fitted shape parameter k̂.
    """
    lw = np.asarray(log_w, dtype=np.float64)
    lw -= lw.max()                     # stabilise
    w  = np.exp(lw)

    n_tail      = max(1, int(len(w) * tail_frac))
    tail_w      = np.sort(w)[-n_tail:]
    shifted     = tail_w - tail_w.min()    # GP must be on (0, ∞)
    k_hat, _, _ = genpareto.fit(shifted)
    return k_hat
