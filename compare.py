import torch

from capsnet import DigitCaps
from demo import procedure1

def main():
    torch.manual_seed(127)

    u = torch.tensor([[[0.5, 1.0, 2.0],
                      [1.0, 1.0, 2.0]]])
    u = torch.tensor([[[0.5, 1.0],
                       [1.0, 1.0],
                       [1.0, 2.0]]])
    print("u size:", u.size())

    batch_size, in_capsules, in_dimension = u.size()
    out_capsules = 2
    out_dimension = 2

    # W, result = procedure1(u, out_capsules, out_dimension, r=3)
    # print("wwwwww", W)

    print("u:", u)
    x = torch.stack([u] * in_capsules, dim=2).unsqueeze(4)
    print("x:", x)

    # from github
    alternative = DigitCaps(out_capsules, in_capsules, in_dimension, out_dimension)

    res_didigtCaps = alternative.forward(u)
    # res2 = result1.squeeze(0)

    print("res digitcaps", res_didigtCaps)
    # print("res2:", res2)

    # my implementation
    W, res_custom = procedure1(u.squeeze(0), out_capsules, out_dimension, W=alternative.W.squeeze(0))
    # assert torch.allclose(result, res2) # strict equal was failing because of precision of floats

    print("res_github:", res_didigtCaps)
    print("res_custom:", res_custom)
    assert torch.allclose(res_didigtCaps, res_custom)



if __name__ == "__main__":
    main()