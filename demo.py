import torch
import torch.nn.functional as F
# capsules at higher level
capsules_layer_0 = torch.randn(3, 2)
capsules_layer_0 = torch.tensor([[0.5, 1],
                                 [1, 1],
                                 [0.5, 0.5]])

capsules_layer_1 = torch.zeros(2, 2)

print("layer_0:", capsules_layer_0, sep="\n")
print("layer_1:", capsules_layer_1, sep="\n")


def routing(r: int=1):
    """Dynamic routing.
    
    """
    # dynamic routing
    i, i_size = capsules_layer_0.size()
    j, j_size = capsules_layer_1.size()

    b_ij = torch.zeros([j, i])

    u_hat = capsules_layer_0#.unsqueeze(0).repeat(2, 1, 1)
    print("u_hat:", u_hat)

    for iter in range(r):
        print("iteration", iter, "=" * 60)

        print("b_ij:", b_ij, sep="\n")

        c_i = F.softmax(b_ij, dim=1)
        c_i = c_i.unsqueeze(0).repeat(2, 1, 1)
        print("c_i:", c_i, sep="\n")
        
        print("c_i", c_i.size())
        print("u_hat", u_hat.size())

        s_ij = c_i.matmul(u_hat.unsqueeze(0).repeat(2, 1, 1)).sum(0)
        print("s_ij:", s_ij)
        
        v_j = squash(s_ij)
        print("v_j:", v_j, sep="\n")
        
        # temp = u_hat.squeeze(1)
        # print("u_hat squeezed", temp)
        b_ij = b_ij + (u_hat.matmul(v_j)).sum(1)

    # return v_j
        


def squash(input_tensor):
    squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
    output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
    return output_tensor


routing()
