import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as td

class AEVB(nn.Module):
    def __init__(self, T, D, K, lambda_h, hidden_size = 128, init_temp = 1.0, device="cpu"):
        super(AEVB, self).__init__()

        self.T = T
        self.D = D
        self.K = K
        self.temp = torch.tensor(init_temp).to(device)
        self.lambda_h = lambda_h

        self.rnn_p = nn.ModuleList([nn.RNN(input_size=1, hidden_size=hidden_size).to(device) for i in range(self.K)])
        self.linear_p = nn.ModuleList([nn.Linear(hidden_size, 1).to(device) for i in range(self.K)]) 

        self.rnn_q = nn.ModuleList([nn.RNN(input_size=self.D, hidden_size=hidden_size).to(device) for i in range(self.K)])
        self.linear_q = nn.ModuleList([nn.Linear(hidden_size, 1).to(device) for i in range(self.K)])

        self.rnn_z = nn.ModuleList([nn.RNN(input_size=self.D, hidden_size=hidden_size).to(device) for i in range(self.K)])
        # self.linear_z = nn.ModuleList([nn.Linear(hidden_size, 1).to(device) for i in range(self.K)])
        self.linear_z_mu = nn.ModuleList([nn.Linear(hidden_size, 1).to(device) for i in range(self.K)])
        self.linear_z_logvar = nn.ModuleList([nn.Linear(hidden_size, 1).to(device) for i in range(self.K)])

        self.rnn_z0 = nn.RNN(input_size=self.D, hidden_size=hidden_size).to(device)
        self.linear_z0_mu = nn.Linear(hidden_size, 1).to(device)
        self.linear_z0_logvar = nn.Linear(hidden_size, 1).to(device)

        self.theta = torch.rand((self.K,  self.D), device=device) ## number of hmm chains, 2 states for each HMM, the dimension of the data at one time point
        self.theta = torch.nn.Parameter(self.theta)

        self.theta0 = torch.randn((1,  self.D), device=device) ## number of hmm chains, 2 states for each HMM, the dimension of the data at one time point
        self.theta0 /= torch.exp(torch.linalg.norm(self.theta0, axis=0))
        self.theta0 = torch.nn.Parameter(self.theta0)

        self.A = torch.randn(self.K, device=device)
        self.A /= torch.exp(torch.linalg.norm(self.A, axis=0))
        self.A = torch.nn.Parameter(self.A)

        self.A0 = torch.randn(1, device=device)
        self.A0 /= torch.exp(torch.linalg.norm(self.A0, axis=0))
        self.A0 = torch.nn.Parameter(self.A0)

        self.sigp = torch.randn((self.K, 1), device=device)
        self.sigp /= torch.exp(torch.linalg.norm(self.sigp, axis=0))
        self.sigp = torch.nn.Parameter(self.sigp)

        self.sigp0 = torch.randn(1, device=device)
        self.sigp0 /= torch.exp(torch.linalg.norm(self.sigp0, axis=0))
        self.sigp0 = torch.nn.Parameter(self.sigp0)
        
        self.device = device

    def encode_p(self, y, k):

      h, _ = self.rnn_p[k](y)
      h = F.relu(h)
      h = self.linear_p[k](h)

      return h #be careful with the dimension
      
    def encode_q(self, y):
      out = []
      for k in range(self.K):
        h, _ = self.rnn_q[k](y)
        h = F.relu(h)
        h = self.linear_q[k](h)
        out.append(h)

      out = torch.stack(out, dim=-1).view(self.T, self.K)  ## check dimension
      return out #be careful with the dimension

    def encode_z(self, y):
        mus = []
        logvars = []
        for k in range(self.K):
            h, _ = self.rnn_z[k](y)
            h = F.relu(h)
            mus.append(self.linear_z_mu[k](h))      # shape (T,1)
            logvars.append(self.linear_z_logvar[k](h))  # shape (T,1)
        return mus, logvars

    def encode_z0(self, y):
        
        h, _ = self.rnn_z0(y)
        h = F.relu(h)
        mu = self.linear_z0_mu(h)
        var = self.linear_z0_logvar(h)
        return mu, var

    def forward(self, y):

        ## get the logit for the relaxedbernoulli
        rho_logit1 = self.encode_q(y)
        q_h = td.relaxed_bernoulli.LogitRelaxedBernoulli(self.temp, logits= rho_logit1)
        h_logit = q_h.rsample()      ## reparameterization for RB
        log_prob_h = q_h.log_prob(h_logit)       ## evaluate the loss

        ## reparameterization for the continuous latent state z
        mus, logvars = self.encode_z(y)
        mu = torch.cat(mus, dim=1)          # shape (T, K)
        logvar = torch.cat(logvars, dim=1)  # shape (T, K)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # reparameterization trick

        # log q(z|x): diagonal normal
        q_z = td.Normal(mu, std)
        log_prob_z = q_z.log_prob(z).sum(dim=1)  # sum over K dims, shape (T,)
        
        ## For baseline noise
        mean0, logvar0 = self.encode_z0(y)
        sigq0 = torch.exp(0.5 * logvar0)  # standard deviation

        q_z0 = td.normal.Normal(mean0, sigq0)
        z0 = q_z0.rsample()
        log_prob_z0 = q_z0.log_prob(z0) ## previously forgot to add this q loss for the baseline noise
        
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
        ## make sure theta is nonnegative
        theta = torch.abs(self.theta)
        theta0 = torch.abs(self.theta0)
        ## make sure sigp's are nonnegative
        sigp = torch.abs(self.sigp)
        sigp0 = torch.abs(self.sigp0)
        ## make sure h is between 0 and 1
        h = torch.sigmoid(h_logit)

        ## calculate mu: mu_k = theta_k * h_k * exp(z_tk)
        mu_approx = h * torch.exp(z) @ theta
        
        ## mu0 from the baseline noise
        mu_approx0 = torch.exp(z0) @ theta0
        
        ## evaluate the p loss: Eq log p = E log Poisson pdf(y|mu) = \sum_D E_q log(y_D|mu_D) = \sum_D mean_T(log(y_D|mu_D))
        p_loss = td.Poisson(mu_approx+mu_approx0).log_prob(y).sum(dim=1).mean()

        ## Eq log q = Eq log q(h) + Eq log z(h)
        q_loss = log_prob_h.sum(dim=1).mean() + log_prob_z.mean() + log_prob_z0.mean()

        ## AR process loss E log N(z_t|Az_t-1, sigma) = sum_k mean(log N(z_tk|Az_t-1k, sigma))
        A = torch.diag(self.A)
        z_loss = td.multivariate_normal.MultivariateNormal(torch.matmul(z[:-1], A), torch.diag(sigp[:, 0])).log_prob(z[1:]).mean()

        ## FHMM loss E log RB(ht|ht-1) = \sum_k mean(RB(htk|h_t-1k))

        temp_loss = torch.zeros(self.K, device=self.device)
        for k in range(self.K):
            temp_loss[k] = td.relaxed_bernoulli.LogitRelaxedBernoulli(self.temp, logits=rho_logit[k][:-1]).log_prob(h_logit[1:, k].view(-1, 1)).mean()
        z0_loss = td.normal.Normal(self.A0 * z0[:-1], sigp0).log_prob(z0[1:]).mean()
        L1_reg = torch.abs(h).sum()/self.T
        delta = torch.abs(z[1:]-z[:-1]).sum(dim=1).mean()
        sigp_loss =  0 #td.gamma.Gamma(1, 10).log_prob(sigp).sum()/self.T
        sigp0_loss = 0 #td.gamma.Gamma(1, 10).log_prob(sigp0).sum()/self.T
        
        total_loss = -p_loss + q_loss - temp_loss.sum() - z_loss.sum() - z0_loss + self.lambda_h * L1_reg - sigp_loss - sigp0_loss + 0 * delta
        
        return total_loss, mu_approx, mu_approx0