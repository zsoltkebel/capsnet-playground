import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def squash(input_tensor):
    squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
    output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
    return output_tensor

# Procedure 1 Routing algorithm.
def procedure1(in_capsules, out_capsule_count, out_capsule_dimension, r: int=3, W=None):
    """Dynamic routing.
    
    Args:
        in_capsules: the matrix of incomding capsules of shape (caps count x caps dimension)
        out_capsule_count: number of capsules in upper level
        out_capsule_dimension: dimension of upper level capsules
        r: number of dynamic routing iterations (defaults to 3)
        W: the weight matrix (defaults to None) if None it is randomly generated
            this should have shape of (in_capsule_count, out_capsule_count, out_capsule_dimension, in_capsule_dimension)
    """
    in_capsule_count, in_capsule_dimension = in_capsules.size()

    if W is None:
        W = torch.randn(in_capsule_count, out_capsule_count, out_capsule_dimension, in_capsule_dimension)

    # line 2: for all capsule i in layer l and capsule j in layer (l+1): bij ← 0.
    b_ij = torch.zeros(in_capsule_count, out_capsule_count)

    logger.debug("Starting dynamic routing [%d x %d] -> [%d x %d] (capsule count x capsule dimension)", in_capsule_count, in_capsule_dimension, out_capsule_count, out_capsule_dimension)

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
    # ^ format incoming capsules into a shape that can be matrix multiplied with the weight
    logger.debug("Incoming capsules shape after adjuting for matrix multiplication by the weight matrix: %s", u.size())

    # Compute predicted outputs (u_hat) using matrix multiplication
    u_hat = torch.matmul(W, u)  # Shape: [3, 2, 2, 1]
    u_hat = u_hat.squeeze(-1)  # Shape: [3, 2, 2] (Remove the last singleton dimension)
    print("u_hat:", u_hat)
    print("u_hat size:", u_hat.size())

    # line 3: for r iterations do
    for iteration in range(r):
        # logger.debug("iteration" + iteration + "=" * 60)

        logger.debug("b_ij:\n%s", b_ij)

        # Compute coupling coefficients
        # line 4: for all capsule i in layer l: c_i ← softmax(b_i) 
        c_ij = F.softmax(b_ij, dim=0) # TODO either dim 0 or 1 0 -> 0.33 values, 1 -> 0.5 values
        c_ij = c_ij.unsqueeze(-1)
        logger.debug("c_ij:\n%s", c_ij)

        logger.debug("c_ij size: %s", c_ij.size())
        logger.debug("u_hat size: %s", u_hat.size())

        # Weighted sum of predictions
        # line 5
        s_j = torch.sum(c_ij * u_hat, dim=0)  # Weighted sum: [n_higher, d_out]
        logger.debug("s_j\n%s", s_j.unsqueeze(2))

        # Squash function
        # line 6
        v_j = squash(s_j.unsqueeze(2))
        logger.debug("v_j:\n%s", v_j)


        v_j_transformed = v_j.squeeze(2).unsqueeze(0).expand(in_capsule_count, -1, -1)

        # print("u_hat", u_hat)
        # print("t", t)

        b_ij = b_ij + (u_hat * v_j_transformed).sum(-1) # sum also squeezes matrix by default

    return W, v_j
    
if __name__ == "__main__":
    # main()
    logging.basicConfig(level=logging.DEBUG)
    u = torch.tensor([
         [0.0, 1.0],
         [1.0, 1.0],
         [2.0, 3.0],
        #  [1.0, 1.0],
        #  [1.0, 1.0],
    ])
    procedure1(u, 2, 2)