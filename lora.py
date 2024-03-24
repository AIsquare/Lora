import torch
import numpy as np
_ = torch.manual_seed(0)

# Generate a rank-deficient matrix W
d, k = 10, 10

W_rank = 2
W= torch.randn(d, W_rank) @ torch.randn(W_rank, k)
#print(W)

# Evaluate the rank of the matrix W
W_rank = np.linalg.matrix_rank(W)
print(f'Rank of the W: {W_rank}')

# Calculate the SVD decompostion of the W matrix
#Perform SVD on W (W = UxSxV^T)
U, S, V = torch.svd(W)

#For rank-r factorization, keep only the first r singular values (and corresponding columns U and V)
U_r = U[:, :W_rank]
#print(U_r.shape)
S_r = torch.diag(S[:W_rank])
V_r = V[:, :W_rank].t()

# Compute B = U_r*S_r and A = V_r
B = U_r @ S_r
A = V_r
print(f'Shape of B: {B.shape}')
print(f'Shape of A: {A.shape}')

#Given the same input, check the output using the original W matrix and the matrices resulting from the decomposition.
bias = torch.randn(d)
x = torch.randn(d)

# Compute y = Wx + bias
y = W @ x + bias
# Compute y' = (B*A)x + bias
y_prime = (B@A)@x + bias
print('Original y using W: \n')
print("")
print("y' computed using BA:\n", y_prime)

print("Total parameters of W: ", W.nelement())
print("Total parameters of B and A: ", B.nelement() + A.nelement())



