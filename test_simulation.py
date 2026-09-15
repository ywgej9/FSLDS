import torch
import numpy as np
from torch import optim
import argparse
import time
import torch.distributions as td
import random
# import wandb

from model_mlp import AEVB   # updated class that supports masks
from data_simulation import simulate_data
from utils import adjust_theta_learning_rate, posterior_predictive_check, psis_khat

def main(args):

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # Simulate some data
    true_y, true_h, true_theta_a, Z, rate = simulate_data(random=False)
    data = torch.tensor(true_y, dtype=torch.float32, device=device) # shape (T,D)
    T, D = data.shape
    T_train=T
    # T_train = 700

    training_data = data[:T_train]
    test_data = data[T_train:]

    # Initialize model
    model = AEVB(T=T_train, D=D, K=args.number_feature, lambda_h=args.lambda_h, device=device)

    # Set up optimizer
    optimizer = optim.Adam([
        {"params": [model.theta], "lr": args.lr, 'name': 'theta'},
        {"params": [p for n, p in model.named_parameters() if n != 'theta'], 
         'lr': args.lr, "name": "other"},
    ], betas=[0.95, 0.99])

    model.train()
    ELBO = []
    start_time = time.time()

    for steps in range(args.n_iter):
        adjust_theta_learning_rate(optimizer, steps, 
                                   theta_initial_lr=0.1, 
                                   theta_lr_after_threshold=0.01, 
                                   epoch_threshold=200)
        optimizer.zero_grad()

        
        (h_logit, log_prob_h, z, log_prob_z, 
         z0, log_prob_z0, rho_logit) = model(training_data)

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
        
        # print(f"Step: {steps}, Loss: {loss.item():.4f}")
        total_grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                gnorm = param.grad.data.norm(2).item()
                total_grad_norm += gnorm**2
        total_grad_norm = total_grad_norm**0.5

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        model.update_temp(steps)
        ELBO.append(loss.item())

        if steps % 100 == 0:
            print(f"Step: {steps}, Loss: {loss.item():.4f}")


    end_time = time.time()
    elapsed_time = end_time - start_time

    T_pred = T - T_train
    model.eval()
    # log_weights = posterior_predictive_check(model, data.to(device), S=1200, device=device)
    k_hat = 0  # psis_khat(log_weights)
    predictions = None

    # with torch.no_grad():
    #     y_pred = torch.zeros(T_pred, D, device=device)
    #     h_pred = torch.zeros(T_pred, model.K, device=device)
    #     z_pred = torch.zeros(T_pred, model.K, device=device)
    #     z0_pred = torch.zeros(T_pred, 1, device=device)
    #     mu_pred = torch.zeros(T_pred, D, device=device)
    #     H_last = torch.zeros(T_pred+1, model.K, device=device)
        
    #     for t in range(T_pred):
            
    #         h_logit_train, _, z_train, _, z0_train, _, _ = model(data[t:(T_train+t)])

    #         h_last  = h_logit_train[-1]   # shape (K,)

    #         z_last  = z_train[-1]                        # shape (K,)
    #         z0_last = z0_train[-1]                       # shape (1,)
            
    #         ## pred z
    #         A      = torch.diag(model.A)                 # AR matrix (K×K, diagonal)
    #         sigp   = torch.abs(model.sigp)[:, 0]         # variance per dim  (K,)
    #         sigp0  = torch.abs(model.sigp0)              # scalar
    #         eps_z  = torch.randn_like(z_last) * torch.sqrt(sigp)  # (K,)
    #         z_t    = (A @ z_last) + eps_z                     # AR(1)

    #         # ---- 2. sample z0_t ---------------------------------------------
    #         eps_z0 = torch.randn_like(z0_last) * sigp0        # (1,)
    #         z0_t   = model.A0 * z0_last + eps_z0              # scalar AR(1)
    #         ## pred h
    #         temp   = model.temp                          # final Gumbel‑Softmax temperature

    #         rho_logits = torch.stack(
    #             [model.encode_p(h_last[k].view(1,1), k).squeeze(0)
    #             for k in range(model.K)]
    #         ).flatten()                            # (K,)

    #         q_h   = td.relaxed_bernoulli.LogitRelaxedBernoulli(
    #                 temp, logits=rho_logits)
    #         h_t   = q_h.rsample() #torch.sigmoid(q_h.sample())              # (K,)

    #         mu_pred0_t = torch.exp(z0_t) @ torch.abs(model.theta0)          # shape (D,)
    #         mu_pred_t = torch.sigmoid(h_t) * torch.exp(z_t) @ torch.abs(model.theta)     # shape (T,D)
    #         mu_pred_t = mu_pred_t + mu_pred0_t
    #         y_pred_t = torch.poisson(mu_pred_t)  # shape (D,)

    #         # ---- 4. store & feed forward ------------------------------------
    #         z_pred[t]   = z_t
    #         z0_pred[t]  = z0_t
    #         h_pred[t]   = h_t
    #         mu_pred[t] = mu_pred_t
    #         y_pred[t] = y_pred_t
    #         H_last[t] = h_last
        
    #     rmse = torch.sqrt(torch.mean((mu_pred - data[T_train:])**2))
    #     pred_rmse = torch.sqrt(torch.mean((y_pred - data[T_train:])**2))                    
            
    #     predictions = {
    #         'y_pred': y_pred,
    #         'h_pred': h_pred,
    #         'z_pred': z_pred,
    #         'z0_pred': z0_pred,
    #         'mu_pred': mu_pred,
    #         "H_last": H_last,
    #         "rmse": rmse,
    #         "pred_rmse": pred_rmse,
    #     }
        
    return (h_logit.detach().cpu(),
            z.detach().cpu(),
            z0.detach().cpu(),
            mu_approx.detach().cpu(),
            mu0_approx.detach().cpu(),
            ELBO,
            model.cpu(),
            data,
            true_h,
            true_theta_a,
            Z,
            rate,
            elapsed_time,
            predictions,
            k_hat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulation data test with holdout mask.")
    parser.add_argument("-n_iter", "--n_iter", default=1000, type=int)
    parser.add_argument("-lr", "--lr", default=1e-2, type=float)
    parser.add_argument("-s", "--seed", default=1, type=int)
    parser.add_argument("-K", "--number-feature", default=2, type=int)
    parser.add_argument("-lambda_h", "--lambda_h", default=0.1, type=float)
    parser.add_argument("-device", type = int, default=0)
    args = parser.parse_args()
    
    # Run training
    (h_logit, z, z0, mu_approx, mu0_approx, 
     ELBO, model_learned, data, 
     true_h, true_theta_a, Z, rate, elapsed_time, predictions, khat) = main(args)

    # Save results
    results = {
        'h_logit': h_logit,
        'z': z,
        'z0': z0,
        'mu_approx': mu_approx,
        'mu0_approx': mu0_approx,
        'ELBO': ELBO,
        'data': data,
        'true_h': true_h,
        'true_theta_a': true_theta_a,
        'Z': Z,
        'rate': rate,
        'time': elapsed_time,
        "predictions": predictions,
        "khat": khat,
    }
    output_path = f"results/simulation_1/output_K{args.number_feature}_seed{args.seed}_lambda{args.lambda_h}.pth"
    torch.save(results, output_path)

    model_path = f"results/simulation_1/model_K{args.number_feature}_seed{args.seed}_lambda{args.lambda_h}.pth"
    torch.save(model_learned.state_dict(), model_path)

    print("Training results and model saved successfully.")
    del model_learned
