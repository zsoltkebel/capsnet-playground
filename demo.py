import torch
import torch.nn.functional as F

# torch.manual_seed(127)

capsules_layer_0 = torch.randn(3, 2)
capsules_layer_0 = torch.tensor([[0.5, 1],
                                [1, 1],
                                [0.5, 0.5]])

capsules_layer_1 = torch.zeros(2, 2)

print("layer_0:", capsules_layer_0, sep="\n")
print("layer_1:", capsules_layer_1, sep="\n")

capsule_count_lower_level, capsule_dimension_lower_level = capsules_layer_0.size()
capsule_count_higher_level, capsule_dimension_higher_level = capsules_layer_1.size()

# Randomised weight matrices
W = torch.randn(capsule_count_lower_level, capsule_count_higher_level, capsule_dimension_higher_level, capsule_dimension_lower_level)

print("W:", W)
print("W size", W.size())

def squash(input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

def main():
    

    # Procedure 1 Routing algorithm.
    def routing(r: int=3):
        """Dynamic routing.
        
        """
        # line 2: for all capsule i in layer l and capsule j in layer (l+1): bij ← 0.
        b_ij = torch.zeros(capsule_count_lower_level, capsule_count_higher_level)

        # Reshaping capsules_layer_0 to match dimensions for multiplication with weight matrix W
        # begins with lower layer sizes
        # [num_of_capsules, capsule_dimension]
        # --unsqueeze(1)-->
        # [num_of_capsules, 1, capsule_dimension]
        # --expand(-1, num_of_capsules_layer_next, -1)-->
        # [num_of_capsules, num_of_capsules_next_layer, capsule_dimension]
        # --unsqueeze(-1)-->
        # [num_of_capsules, num_of_capsules_next_layer, capsule_dimension, 1]
        u = (
            capsules_layer_0
            .unsqueeze(1) # add a new dimension at index 1
            .expand(-1, capsule_count_higher_level, -1)
            .unsqueeze(-1) # add a new dimension after the current last
        )
        print("u size:", u.size())
        print("u after unsqueeze:", u)

        # Compute predicted outputs (u_hat) using matrix multiplication
        u_hat = torch.matmul(W, u)  # Shape: [3, 2, 2, 1]
        u_hat = u_hat.squeeze(-1)  # Shape: [3, 2, 2] (Remove the last singleton dimension)
        print("u_hat:", u_hat)
        print("u_hat size:", u_hat.size())

        # line 3: for r iterations do
        for iter in range(r):
            print("iteration", iter, "=" * 60)

            print("b_ij:", b_ij, sep="\n")

            # Compute coupling coefficients
            # line 4: for all capsule i in layer l: c_i ← softmax(b_i) 
            c_ij = F.softmax(b_ij, dim=0)
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
            #TODO this should be good now
            # print("b_ij size:", b_ij.size())
            # print("u_hat size:", u_hat.size())
            # print("v_j size:", v_j.size())

            t = v_j.unsqueeze(0).expand(3, -1, -1)

            # print("u_hat", u_hat)
            # print("t", t)

            b_ij = b_ij + (u_hat * t).sum(-1) # sum also squeezes matrix by default

        return v_j


    routing()


# Procedure 1 Routing algorithm.
def procedure1(in_capsules, out_capsule_count, out_capsule_dimension, r: int=3, W=None):
    """Dynamic routing.
    
    """
    in_capsule_count, in_capsule_dimension = in_capsules.size()

    if W is None:
        W = torch.randn(in_capsule_count, out_capsule_count, out_capsule_dimension, in_capsule_dimension)

    # line 2: for all capsule i in layer l and capsule j in layer (l+1): bij ← 0.
    b_ij = torch.zeros(in_capsule_count, out_capsule_count)

    # Reshaping capsules_layer_0 to match dimensions for multiplication with weight matrix W
    # begins with lower layer sizes
    # [num_of_capsules, capsule_dimension]
    # --unsqueeze(1)-->
    # [num_of_capsules, 1, capsule_dimension]
    # --expand(-1, num_of_capsules_layer_next, -1)-->
    # [num_of_capsules, num_of_capsules_next_layer, capsule_dimension]
    # --unsqueeze(-1)-->
    # [num_of_capsules, num_of_capsules_next_layer, capsule_dimension, 1]
    u = (
        in_capsules
        .unsqueeze(1) # add a new dimension at index 1
        .expand(-1, out_capsule_count, -1)
        .unsqueeze(-1) # add a new dimension after the current last
    )
    print("u size:", u.size())
    print("u after unsqueeze:", u)

    # Compute predicted outputs (u_hat) using matrix multiplication
    u_hat = torch.matmul(W, u)  # Shape: [3, 2, 2, 1]
    u_hat = u_hat.squeeze(-1)  # Shape: [3, 2, 2] (Remove the last singleton dimension)
    print("u_hat:", u_hat)
    print("u_hat size:", u_hat.size())

    # line 3: for r iterations do
    for iteration in range(r):
        print("iteration", iteration, "=" * 60)

        print("b_ij:", b_ij, sep="\n")

        # Compute coupling coefficients
        # line 4: for all capsule i in layer l: c_i ← softmax(b_i) 
        c_ij = F.softmax(b_ij, dim=0) # TODO either dim 0 or 1 0 -> 0.33 values, 1 -> 0.5 values
        c_ij = c_ij.unsqueeze(-1)
        print("c_ij:", c_ij, sep="\n")

        print("c_ij size", c_ij.size())
        print("u_hat size", u_hat.size())

        # Weighted sum of predictions
        # line 5
        s_j = torch.sum(c_ij * u_hat, dim=0)  # Weighted sum: [n_higher, d_out]
        print("s_j", s_j.unsqueeze(2))

        # Squash function
        # line 6
        v_j = squash(s_j.unsqueeze(2))
        print("v_j:", v_j, sep="\n")


        v_j_transformed = v_j.squeeze(2).unsqueeze(0).expand(in_capsule_count, -1, -1)

        # print("u_hat", u_hat)
        # print("t", t)

        b_ij = b_ij + (u_hat * v_j_transformed).sum(-1) # sum also squeezes matrix by default

    return W, v_j
    
if __name__ == "__main__":
    # main()
    u = torch.tensor([
         [0.0, 1.0],
         [1.0, 1.0],
         [2.0, 3.0],
         [1.0, 1.0],
         [1.0, 1.0],
    ])
    procedure1(u, 2, 2)