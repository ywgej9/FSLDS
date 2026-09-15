import torch
import numpy as np
from torch import optim
# from model_holdout import AEVB
from model_rnn import AEVB
import argparse
import time
import torch.distributions as td
import random
from utils import adjust_theta_learning_rate
from scipy.stats import genpareto         

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

def main(args):
    device = args.device
    data = torch.from_numpy(np.load(args.file_path)).float().to(device)  ## .T for the GBZ data
    T, D = data.shape
    T_train = int(T * 3/4 + 100) # 3/4 of the data for training
    # T_train = T
    training_data = data[:T_train]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = AEVB(T=T_train, D = D, K = args.number_feature, lambda_h = args.lambda_h, device=device)

    optimizer = optim.Adam([
    {"params": [model.theta], "lr":args.lr, 'name': 'theta'},
    {"params": [p for n, p in model.named_parameters() if n != 'theta'], 'lr': args.lr, "name": "other parameters"}, # 1e-3 for real data
],
                       betas=[0.95, 0.99])

    model.train()
    ELBO = []
    start_time = time.time()
    for steps in range(args.n_iter):
        adjust_theta_learning_rate(optimizer, steps, theta_initial_lr=0.1, theta_lr_after_threshold=0.001, epoch_threshold=200) ## may need to change this later
        optimizer.zero_grad()
        (h_logit, log_prob_h, z, log_prob_z, z0, log_prob_z0, rho_logit) = model(training_data)

        # Compute loss
        loss, mu_approx, mu0_approx = model.loss(
            y=training_data,
            h_logit=h_logit,
            log_prob_h=log_prob_h,
            z=z,
            log_prob_z=log_prob_z,
            z0=z0,
            log_prob_z0=log_prob_z0,
            rho_logit=rho_logit
        )

        loss.backward()
        ELBO.append(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        # scheduler.step()
        model.update_temp(steps)
        if steps % 100 == 0:
            print('step: ', steps, 'loss: ', loss.item())
    end_time = time.time()
    elapsed_time = end_time - start_time
    model.eval()
    # log_weights = posterior_predictive_check(model, data.to(device), S=1400, device=device)
    k_hat = 0 #psis_khat(log_weights)
    predictions = None
    T_pred = T - T_train
    model.eval()
    with torch.no_grad():
        y_pred = torch.zeros(T_pred, D, device=device)
        h_pred = torch.zeros(T_pred, model.K, device=device)
        z_pred = torch.zeros(T_pred, model.K, device=device)
        z0_pred = torch.zeros(T_pred, 1, device=device)
        mu_pred = torch.zeros(T_pred, D, device=device)

        
        for t in range(T_pred):
            window = data[t:(T_train+t)]
            
            h_logit_train, _, z_train, _, z0_train, _, _ = model(window)

            h_last  = h_logit_train[-1]   # shape (K,)

            z_last  = z_train[-1]                        # shape (K,)
            z0_last = z0_train[-1]                       # shape (1,)
            
            ## pred z
            A      = torch.diag(model.A)                 # AR matrix (K×K, diagonal)
            sigp   = torch.abs(model.sigp)[:, 0]         # std‑dev per dim  (K,)
            sigp0  = torch.abs(model.sigp0)              # scalar
            eps_z  = torch.randn_like(z_last) * sigp          # (K,)
            z_t    = (A @ z_last) + eps_z                     # AR(1)

            # ---- 2. sample z0_t ---------------------------------------------
            eps_z0 = torch.randn_like(z0_last) * sigp0        # (1,)
            z0_t   = model.A0 * z0_last + eps_z0              # scalar AR(1)
            ## pred h
            temp   = model.temp                          # final Gumbel‑Softmax temperature

            rho_logits = torch.stack(
                [model.encode_p(h_last[k].view(1,1), k).squeeze(0)
                for k in range(model.K)]
            ).flatten()                            # (K,)

            q_h   = td.relaxed_bernoulli.LogitRelaxedBernoulli(
                    temp, logits=rho_logits)
            h_t   = q_h.rsample() #torch.sigmoid(q_h.sample())              # (K,)

            mu_pred_t = torch.sigmoid(h_t) * torch.exp(z_t) @ torch.abs(model.theta)     # shape (D,)
            mu_pred0_t = torch.exp(z0_t) @ torch.abs(model.theta0)          # shape (D,)
            mu_pred_t = mu_pred_t + mu_pred0_t
            y_pred_t = torch.poisson(mu_pred_t)  # shape (D,)

            # ---- 4. store & feed forward ------------------------------------
            z_pred[t]  = z_t
            z0_pred[t] = z0_t
            h_pred[t]  = h_t
            mu_pred[t] = mu_pred_t
            y_pred[t] = y_pred_t
        pred_rmse = torch.sqrt(torch.mean((y_pred - data[T_train:])**2))
        rmse = torch.sqrt(torch.mean((mu_pred - data[T_train:])**2))
        mae = torch.abs(y_pred - data[T_train:]).mean() 
            
        predictions = {
            'y_pred': y_pred,
            'h_pred': h_pred,
            'z_pred': z_pred,
            'z0_pred': z0_pred,
            'mu_pred': mu_pred,
            "pred_rmse": pred_rmse,
            'rmse': rmse,
            'mae': mae,
        }
        

    return h_logit.detach().cpu(), z.detach().cpu(), z0.detach().cpu(), mu_approx.detach().cpu(), mu0_approx.detach().cpu(), ELBO, model.cpu(), elapsed_time, k_hat, predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real data test"
    )
    parser.add_argument("-n_iter", "--n_iter", default=1000, type=int)
    parser.add_argument("-lr", "--lr", default=1e-3, type=float)
    parser.add_argument("-s", "--seed", default=1, type=int)
    parser.add_argument("-lambda_h", "--lambda_h", default=0.1, type=float)
    parser.add_argument("-K", "--number-feature", default=10, type=int)
    parser.add_argument("-file", "--file-path", default="4weekdata/p1a2_4week.npy", type=str)
    parser.add_argument("-device", type = int, default=0)
    args = parser.parse_args()
    
    h_logit, z, z0, mu_approx, mu0_approx, ELBO, model_learned, elapsed_time, khat, predictions = main(args)
    results = {
    'h_logit': h_logit,
    'z': z,
    'z0': z0,
    'mu_approx': mu_approx,
    "mu0_approx": mu0_approx,
    "ELBO": ELBO,
    "time": elapsed_time,
    "predictions": predictions,
    "khat": khat
}
    file_name = args.file_path.split("/")[1].split(".")[0]
    # file_name = args.file_path.split(".")[0]

    # output_path = f"results/output_{file_name}_K{args.number_feature}_seed{args.seed}_lambda{args.lambda_h}.pth"
    # output_path = f"results/newdata/output_{file_name}_K{args.number_feature}_seed{args.seed}_lambda{args.lambda_h}.pth"

    torch.save(results, output_path)

    # Save the model's state dictionary

    # model_path = f"results/model_{file_name}_K{args.number_feature}_seed{args.seed}_lambda{args.lambda_h}.pth"
    # model_path = f"results/newdata/model_{file_name}_K{args.number_feature}_seed{args.seed}_lambda{args.lambda_h}.pth"

    torch.save(model_learned.cpu().state_dict(), model_path)
    
    print("Training results and model saved successfully.")
del model_learned