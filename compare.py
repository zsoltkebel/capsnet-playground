import torch

from test import DigitCaps
from demo import procedure1

def main():
    torch.manual_seed(127)

    u = torch.tensor([[0.5, 1.0],
                      [1.0, 1.0],
                      [0.5, 0.5],
                      [2.0, 0.0]])
    
    in_capsules, in_dimension = u.size()
    out_capsules = 2
    out_dimension = 2

    W, result = procedure1(u, out_capsules, out_dimension, r=3)
    # print("wwwwww", W)

    alternative = DigitCaps(out_capsules, in_capsules, in_dimension, out_dimension)
    alternative.W = W.unsqueeze(0)
    
    result1 = alternative.forward(u.unsqueeze(0))
    res2 = result1.squeeze(0)

    print("res1:", result)
    print("res2:", res2)

    assert torch.allclose(result, res2) # strict equal was failing because of precision of floats



if __name__ == "__main__":
    main()