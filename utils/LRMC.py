import torch
import copy


def svt(mat, tau):
    u, s, v = torch.linalg.svd(mat, full_matrices = False)
    idx = torch.sum(s > tau)
    return u[:, : idx] @ torch.diag(s[: idx] - tau) @ v[: idx, :]

def lrmc_imputer(mat, rho0, epsilon, maxiter):
    dim1, dim2 = mat.shape
    pos_missing = torch.where(mat == 0)
    last_mat = copy.deepcopy(mat)
    snorm = torch.linalg.norm(mat, 'fro')
    T = torch.zeros(dim1, dim2).cuda()
    Z = copy.deepcopy(mat)

    Z[pos_missing] = torch.mean(Z[Z != 0])
    it = 0
    rho = rho0
    while True:
        rho = min(rho * 1.05, 1e5)
        X = svt(Z - T / rho, 1 / rho)
        Z[pos_missing] = (X + T / rho)[pos_missing]
        T = T + rho * (X - Z)
        tol = torch.linalg.norm((X - last_mat), 'fro') / snorm
        last_mat = copy.deepcopy(X)
        it += 1
        # if it % 1 == 0:
        #     print('Iter: {}'.format(it))
        #     print('Tolerance: {:.6}'.format(tol))
        #     print()
        if (tol < epsilon) or (it >= maxiter):
            break
    return X

def start_imputer(x, mask, mean_std):
    #x = (x * mean_std[1] + mean_std[0]) * mask
    x = x
    rho = 1e-2
    epsilon = 1e-4
    maxiter = 50
    mat_hat = lrmc_imputer(x, rho, epsilon, maxiter)
    x = x * mask + (1-mask) * mat_hat
    return x
    #return (x - mean_std[0]) / mean_std[1]

