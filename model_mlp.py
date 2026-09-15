import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as td

class AEVB(nn.Module):
    def __init__(self, T, D, K, lambda_h, hidden_size=128, init_temp=1.0, device="cpu"):
        super(AEVB, self).__init__()

        self.T = T
        self.D = D
        self.K = K
        self.lambda_h = lambda_h
        self.temp = torch.tensor(init_temp).to(device)

        # MLP for p: input size = 1
        self.mlp_p = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU()
            ).to(device) for _ in range(self.K)
        ])
        self.linear_p = nn.ModuleList([
            nn.Linear(hidden_size, 1).to(device) for _ in range(self.K)
        ])

        # MLP for q: input size = D
        self.mlp_q = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.D, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU()
            ).to(device) for _ in range(self.K)
        ])
        self.linear_q = nn.ModuleList([
            nn.Linear(hidden_size, 1).to(device) for _ in range(self.K)
        ])

        # MLP for z: input size = D
        self.mlp_z = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.D, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU()
            ).to(device) for _ in range(self.K)
        ])
        self.linear_z = nn.ModuleList([
            nn.Linear(hidden_size, 1).to(device) for _ in range(self.K)
        ])
        self.linear_z_logvar = nn.ModuleList([
            nn.Linear(hidden_size, 1).to(device) for _ in range(self.K)
        ])
        # MLP for z0: input size = D
        self.mlp_z0 = nn.Sequential(
            nn.Linear(self.D, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        ).to(device)
        self.linear_z0 = nn.Linear(hidden_size, 1).to(device)

        # Parameters
        self.theta = torch.rand((self.K, self.D), device=device)
        self.theta = torch.nn.Parameter(self.theta)

        self.theta0 = torch.randn((1, self.D), device=device)
        self.theta0 /= torch.exp(torch.linalg.norm(self.theta0, dim=0))
        self.theta0 = torch.nn.Parameter(self.theta0)

        self.A = torch.randn(self.K, device=device)
        self.A /= torch.exp(torch.linalg.norm(self.A, dim=0))
        self.A = torch.nn.Parameter(self.A)

        self.A0 = torch.randn(1, device=device)
        self.A0 /= torch.exp(torch.linalg.norm(self.A0, dim=0))
        self.A0 = torch.nn.Parameter(self.A0)

        self.sigq0 = torch.randn(1, device=device)
        self.sigq0 /= torch.exp(torch.linalg.norm(self.sigq0, dim=0))
        self.sigq0 = torch.nn.Parameter(self.sigq0)

        self.sigp = torch.randn((self.K, 1), device=device)
        self.sigp /= torch.exp(torch.linalg.norm(self.sigp, dim=0))
        self.sigp = torch.nn.Parameter(self.sigp)

        self.sigp0 = torch.randn(1, device=device)
        self.sigp0 /= torch.exp(torch.linalg.norm(self.sigp0, dim=0))
        self.sigp0 = torch.nn.Parameter(self.sigp0)

        self.device = device

    def encode_p(self, y, k):
        """
        MLP for p(h_t | h_{t-1}). Input y is shape (T,1).
        """
        h = self.mlp_p[k](y)
        h = self.linear_p[k](h)
        return h

    def encode_q(self, y):
        """
        MLP for q(h). Input y is shape (T, D), 
        where missing/held-out entries are zeroed out.
        """
        out = []
        for k in range(self.K):
            h = self.mlp_q[k](y)
            h = self.linear_q[k](h)
            out.append(h)
        # stack along last dim => shape (T, K)
        out = torch.stack(out, dim=-1).view(self.T, self.K)
        return out
    
    def encode_z(self, y):
        """
        MODIFIED: Encode both mean and log-variance for z
        """
        out_mean = []
        out_logvar = []
        
        for k in range(self.K):
            h = self.mlp_z[k](y)  # Shared MLP features
            mean_k = self.linear_z[k](h)  # Mean head
            logvar_k = self.linear_z_logvar[k](h)  # Log-variance head
            
            out_mean.append(mean_k)
            out_logvar.append(logvar_k)
        
        mean = torch.concat(out_mean, dim=1)  # (T, K)
        logvar = torch.concat(out_logvar, dim=1)  # (T, K)
        
        return mean, logvar


    def encode_z0(self, y):
        """
        MLP for z0. Input y is shape (T, D).
        """
        h = self.mlp_z0(y)
        mean = self.linear_z0(h)  # shape (T,1)
        return mean

    
    def forward(self, y):
        """
        y: shape (T, D)
        """

        # 1) encode h (Bernoulli latents)
        rho_logit1 = self.encode_q(y)

        rho_logit1 = torch.clamp(rho_logit1, max=20.0)

        q_h = td.relaxed_bernoulli.LogitRelaxedBernoulli(self.temp, logits=rho_logit1)
        h_logit = q_h.rsample()  # reparam sample
        log_prob_h = q_h.log_prob(h_logit)  # shape (T,K)

        # 2) encode z (continuous latents)
        mean, logvar = self.encode_z(y)  # (T, K), (T, K)
        std = torch.exp(0.5 * logvar)  # σ = exp(log(σ²)/2)
        
        q_z = td.Independent(td.Normal(mean, std), 1)  # (T, K) independent normals
        z = q_z.rsample()  # (T, K)
        log_prob_z = q_z.log_prob(z)  # (T,)
        
        # 3) encode z0 (baseline continuous latents)
        mean0 = self.encode_z0(y)
        sigq0 = torch.sqrt(self.sigq0**2)  # scalar or shape (1,)
        q_z0 = td.normal.Normal(mean0, sigq0)  # elementwise normal
        z0 = q_z0.rsample()               # (T,1)
        log_prob_z0 = q_z0.log_prob(z0)   # (T,1)
        
        rho_logit = []
        for k in range(self.K):
            # shape (T,1)
            h_tm1 = h_logit[:, k].view(-1, 1)  # "logit" from previous time
            # input size=1 for MLP
            rho_logit_k = self.encode_p(h_tm1, k)
            rho_logit.append(rho_logit_k)

        return h_logit, log_prob_h, z, log_prob_z, z0, log_prob_z0, rho_logit           
            
    def update_temp(self, steps, ANNEAL_RATE=0.00003, MIN_TEMP=0.1):
        """anneal the temperature"""
        if steps % 10 == 0:
            decay = torch.exp(
                torch.tensor(-ANNEAL_RATE * steps, device=self.device)
            )
            self.temp = torch.clamp(self.temp * decay, min=MIN_TEMP)

    def loss(self, y, h_logit, log_prob_h, z, log_prob_z, z0, log_prob_z0, rho_logit):
        """
        y: (T,D)
        """
        # Force positivity on certain parameters
        theta = torch.abs(self.theta)
        theta0 = torch.abs(self.theta0)
        sigp = torch.abs(self.sigp)
        sigp0 = torch.abs(self.sigp0)

        # Convert h_logit => [0,1]
        h = torch.sigmoid(h_logit)   # shape (T,K)

        # 1) Mean of Poisson from latents (K-component)

        # mu_approx = h * (0.0+torch.exp(z)) @ theta     # shape (T,D)
        mu_approx = h * (0.0+torch.exp(z)) @ theta
        mu_approx0 = torch.exp(z0) @ theta0      # shape (T,D)
        mu_total = mu_approx + mu_approx0
        # 2) Poisson log-prob.
        #    Poisson(...) returns shape (T,D) for .log_prob(y).
        p_log_prob = td.Poisson(mu_total).log_prob(y)
        p_loss = p_log_prob.sum(dim=1).mean()

        # 3) Summation of q(h) and q(z,z0)
        q_loss = log_prob_h.sum(dim=1).mean() + log_prob_z.mean() + log_prob_z0.mean()

        # 4) AR process on z
        #    z[t] ~ N(A*z[t-1], sigp), treat each dimension independently
        A = torch.diag(self.A)
        # shape (T-1,K)
        z_t_minus_1 = z[:-1]  
        z_t = z[1:]
        # Create a multivariate normal for each t
        # We'll interpret diagonal covariance from sigp
        z_loss_dist = td.multivariate_normal.MultivariateNormal(
            z_t_minus_1 @ A, torch.diag(sigp[:, 0])
        )
        z_loss = z_loss_dist.log_prob(z_t).mean()

        # 5) Bernoulli AR for h
        #    h[t] ~ Bernoulli( encode_p(h[t-1]) ), relaxed

        temp_loss = 0.
        for k in range(self.K):
            # shape (T-1,1) for logits => compare with h_logit[1:, k]
            dist_h = td.relaxed_bernoulli.LogitRelaxedBernoulli(
                self.temp, logits=rho_logit[k][:-1]
            )
            temp_loss_k = dist_h.log_prob(h_logit[1:, k].view(-1,1)).mean()
            temp_loss += temp_loss_k
        # 6) AR for z0
        z0_loss_dist = td.normal.Normal(self.A0 * z0[:-1], sigp0)
        z0_loss = z0_loss_dist.log_prob(z0[1:]).mean()

        # 7) Additional regularizers:
        #    L1_reg on h, for instance
        L1_reg = torch.abs(h).sum() / self.T

        sigp0_loss = 0 #td.gamma.Gamma(1, 10).log_prob(sigp0).sum() / self.T
        sigp_loss = 0 #prior_sigp.log_prob(sigp).sum()
        total_loss = (
            -p_loss
            + q_loss
            - temp_loss
            - z_loss
            - z0_loss
            + self.lambda_h * L1_reg
            - sigp_loss
            - sigp0_loss
        )
        return total_loss, mu_approx, mu_approx0
