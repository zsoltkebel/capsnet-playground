import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

class Squash(nn.Module):
    """
    ## Squash

    This is **squashing** function from paper, given by equation $(1)$.

    $$\mathbf{v}_j = \frac{{\lVert \mathbf{s}_j \rVert}^2}{1 + {\lVert \mathbf{s}_j \rVert}^2}
     \frac{\mathbf{s}_j}{\lVert \mathbf{s}_j \rVert}$$

    $\frac{\mathbf{s}_j}{\lVert \mathbf{s}_j \rVert}$
    normalizes the length of all the capsules, whilst
    $\frac{{\lVert \mathbf{s}_j \rVert}^2}{1 + {\lVert \mathbf{s}_j \rVert}^2}$
    shrinks the capsules that have a length smaller than one .
    """

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, s: torch.Tensor):
        """
        The shape of `s` is `[batch_size, n_capsules, n_features]`
        """

        # ${\lVert \mathbf{s}_j \rVert}^2$
        s2 = (s ** 2).sum(dim=-1, keepdims=True)

        # We add an epsilon when calculating $\lVert \mathbf{s}_j \rVert$ to make sure it doesn't become zero.
        # If this becomes zero it starts giving out `nan` values and training fails.
        # $$\mathbf{v}_j = \frac{{\lVert \mathbf{s}_j \rVert}^2}{1 + {\lVert \mathbf{s}_j \rVert}^2}
        # \frac{\mathbf{s}_j}{\sqrt{{\lVert \mathbf{s}_j \rVert}^2 + \epsilon}}$$
        return (s2 / (1 + s2)) * (s / torch.sqrt(s2 + self.epsilon))
    
class Router(nn.Module):
    """
    ## Routing Algorithm

    This is the routing mechanism described in the paper.
    You can use multiple routing layers in your models.

    This combines calculating $\mathbf{s}_j$ for this layer and
    the routing algorithm described in *Procedure 1*.
    """

    def __init__(self, in_caps: int, out_caps: int, in_d: int, out_d: int, iterations: int):
        """
        `in_caps` is the number of capsules, and `in_d` is the number of features per capsule from the layer below.
        `out_caps` and `out_d` are the same for this layer.

        `iterations` is the number of routing iterations, symbolized by $r$ in the paper.
        """
        super().__init__()
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.iterations = iterations
        self.softmax = nn.Softmax(dim=1)
        self.squash = Squash()

        # This is the weight matrix $\mathbf{W}_{ij}$. It maps each capsule in the
        # lower layer to each capsule in this layer
        self.weight = nn.Parameter(torch.randn(in_caps, out_caps, in_d, out_d), requires_grad=True)

    def forward(self, u: torch.Tensor):
        """
        The shape of `u` is `[batch_size, n_capsules, n_features]`.
        These are the capsules from the lower layer.
        """

        # $$\hat{\mathbf{u}}_{j|i} = \mathbf{W}_{ij} \mathbf{u}_i$$
        # Here $j$ is used to index capsules in this layer, whilst $i$ is
        # used to index capsules in the layer below (previous).
        u_hat = torch.einsum('ijnm,bin->bijm', self.weight, u)

        # Initial logits $b_{ij}$ are the log prior probabilities that capsule $i$
        # should be coupled with $j$.
        # We initialize these at zero
        b = u.new_zeros(u.shape[0], self.in_caps, self.out_caps)

        v = None

        # Iterate
        for i in range(self.iterations):
            # routing softmax $$c_{ij} = \frac{\exp({b_{ij}})}{\sum_k\exp({b_{ik}})}$$
            c = self.softmax(b)
            # $$\mathbf{s}_j = \sum_i{c_{ij} \hat{\mathbf{u}}_{j|i}}$$
            s = torch.einsum('bij,bijm->bjm', c, u_hat)
            # $$\mathbf{v}_j = squash(\mathbf{s}_j)$$
            v = self.squash(s)
            # $$a_{ij} = \mathbf{v}_j \cdot \hat{\mathbf{u}}_{j|i}$$
            a = torch.einsum('bjm,bijm->bij', v, u_hat)
            # $$b_{ij} \gets b_{ij} + \mathbf{v}_j \cdot \hat{\mathbf{u}}_{j|i}$$
            b = b + a

        return v
    
torch.manual_seed(127)
test = Router(3, 2, 2, 2, 3)
test_input = torch.tensor([[[0.5, 1],
                                 [1, 1],
                                 [0.5, 0.5]]])
output = test.forward(test_input)

print(output)