import numpy as np
from utils import trans_mat, generate_2d_ar1_process

def simulate_data(T=1000, D=16, K=4, random=False, seed=777533):
    """ Z: D * T
    """
    if random:
        np.random.seed(seed)
        transition_matrix_a = trans_mat(0.01)
        transition_matrix_b = trans_mat(0.015)
        true_h = np.zeros((T, K)).astype(int)

    else:

        true_h = np.zeros((T, K))
        # true_h[500:640, 0] = 1
        # true_h[150:350, 0] = 1
        # true_h[790:920, 0] = 1
        # true_h[170:240, 1] = 1
        # true_h[300:450, 1] = 1
        # true_h[620:730, 1] = 1
        true_h[570:640, 0] = 1
        true_h[150:350, 0] = 1
        true_h[790:960, 0] = 1

        true_h[170:240, 1] = 1
        true_h[300:450, 1] = 1
        true_h[620:730, 1] = 1

        true_h[50:120, 2] = 1
        true_h[180:230, 2] = 1
        true_h[820:890, 2] = 1

        true_h[100:210, 3] = 1
        true_h[330:500, 3] = 1
        true_h[950:995, 3] = 1
        
        true_h = true_h.astype(int)
        t = np.linspace(0, 10, T)
        true_z = np.zeros((T, K))
        true_z[:, 0] = np.cos(t*3) # 1.5
        true_z[:, 2] = 2*np.sin(t*2.5)
        true_z[:, 1] = 0.5 * t - 2
        true_z[:, 3] = 1.5
        Z = true_z + 4
        
        # alpha = [0.2] * K# np.random.normal(0, 1, size=(K))
        # beta = [0.8]*K #np.random.normal(0, 1, size=(K))
        # true_z = generate_2d_ar1_process(alpha, beta, T, K, seed=1337, sigma=0.2)
        # Z = np.exp(true_z) #true_z + 10
        
    true_theta_a = 2*np.array([[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1], [0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0],
                               [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]) + 0.1#np.abs(np.random.random((K, D)))*5#np.array([[0, 2, 4], [4, 5, 3]])#/5 ## dim K by D

    true_y_k = np.zeros((K, T, D))

    rate = np.zeros((K, T, D))
    for k in range(K):
        for i in range(T):
            rate[k, i, :] = true_theta_a[k] * true_h[i, k] * Z[i, k]
        true_y_k[k, :, :] = np.random.poisson(rate[k, :, :])
        # for i in range(T):
        #     true_y_k[k, i, :] = np.random.poisson(rate[k, i]) ## emission from the first mm
    
    true_y = np.sum(true_y_k, axis=0)

    return true_y, true_h, true_theta_a, Z, rate