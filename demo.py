import torch
import torch.nn.functional as F

torch.manual_seed(127)

capsules_layer_0 = torch.randn(3, 2)
capsules_layer_0 = torch.tensor([[0.5, 1],
                                 [1, 1],
                                 [0.5, 0.5]])

capsules_layer_1 = torch.zeros(2, 2)

print("layer_0:", capsules_layer_0, sep="\n")
print("layer_1:", capsules_layer_1, sep="\n")

num_of_capsules_layer_i, capsule_size_layer_i = capsules_layer_0.size()
num_of_capsules_layer_j, capsule_size_layer_j = capsules_layer_1.size()

W = torch.randn(num_of_capsules_layer_i, num_of_capsules_layer_j, capsule_size_layer_j, capsule_size_layer_i)

print("W:", W)
print("W size", W.size())

# Procedure 1 Routing algorithm.
def routing(r: int=3):
    """Dynamic routing.
    
    """
    # line 2: for all capsule i in layer l and capsule j in layer (l+1): bij ← 0.
    b_ij = torch.zeros(num_of_capsules_layer_i, num_of_capsules_layer_j)

    # Reshaping capsules_layer_0 to match dimensions for multiplication
    u = capsules_layer_0.unsqueeze(1).expand(-1, num_of_capsules_layer_j, -1)
    print("u (expanded):", u)
    print("u size:", u.size())

    # Compute predicted outputs (u_hat) using matrix multiplication
    u_hat = torch.matmul(W, u.unsqueeze(-1))  # Shape: [3, 2, 2, 1]
    u_hat = u_hat.squeeze(-1)  # Shape: [3, 2, 2] (Remove the last singleton dimension)
    print("u_hat:", u_hat)
    print("u_hat size:", u_hat.size())

    # line 3: for r iterations do
    for iter in range(r):
        print("iteration", iter, "=" * 60)

        print("b_ij:", b_ij, sep="\n")

        # Compute coupling coefficients
        # line 4: for all capsule i in layer l: c_i ← softmax(b_i) 
        c_ij = F.softmax(b_ij, dim=1)
        c_ij = c_ij.unsqueeze(-1)
        print("c_ij:", c_ij, sep="\n")

        print("c_ij size", c_ij.size())
        print("u_hat size", u_hat.size())

        # Weighted sum of predictions
        # line 5
        s_j = torch.sum(c_ij * u_hat, dim=0)  # Weighted sum: [n_higher, d_out]
        print("s_j", s_j)

        # Squash function
        # line 6
        v_j = squash(s_j)
        print("v_j:", v_j, sep="\n")

        # Update logits
        # line 7
        b_ij = b_ij + (u_hat.matmul(v_j)).sum(1)

    return v_j
        


def squash(input_tensor):
    squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
    output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
    return output_tensor


routing()
