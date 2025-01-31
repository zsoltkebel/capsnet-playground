import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from demo import W as demo_w

USE_CUDA = True if torch.cuda.is_available() else False

class DigitCaps(nn.Module):
    def __init__(self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16, W=None):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        # self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))
        self.W = torch.randn(1, num_routes, num_capsules, out_channels, in_channels)
        print("W in digitcaps:", self.W)
        # self.W = W # demo_w.unsqueeze(0)
        # print(demo_w.unsqueeze(0).size())


    def forward(self, x):
        safe = x[0]
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)
        print("x size:", x.size())
        print("x:", x)

        W = torch.cat([self.W] * batch_size, dim=0)
        print("W:", W)
        # print("x size:", x.squeeze().size())
        # print("x:", x)
        # u = (
        #     safe
        #     .unsqueeze(1) # add a new dimension at index 1
        #     .expand(-1, self.num_capsules, -1)
        #     .unsqueeze(-1) # add a new dimension after the current last
        # )
        # print("u:", u)
        u_hat = torch.matmul(W, x)

        b_ij = torch.zeros(1, self.num_routes, self.num_capsules, 1)
        # b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        # if USE_CUDA:
        #     b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            print("Iteration " + str(iteration) + "=" * 60)
            print("b_ij size", b_ij.size())
            print("b_ij", b_ij)
            c_ij = F.softmax(b_ij, dim=1)
            print("c_ij size", c_ij.size())
            print("c_ij", c_ij)

            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)
            # print("c_ij", c_ij)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            print("s_j:", s_j)

            v_j = self.squash(s_j)
            print("v_j:", v_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

def main():
    torch.manual_seed(127)

    test = DigitCaps(2, 3, 2, 2)
    test_input = torch.tensor([[[0.5, 1],
                                    [1, 1],
                                    [0.5, 0.5]]])

    output = test.forward(test_input)
    print("from other", output)

if __name__ == "__main__":
    main()