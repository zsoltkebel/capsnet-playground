import logging
import unittest

import torch

from capsnet import DigitCaps
from dynamic_routing import procedure1


class CapsuleNetworkTests(unittest.TestCase):

    # def test_squash(self):
    #     u = torch.tensor([
    #         [0.135734 , 1.4105719],
    #         [2.3560739, 0.600849 ]
    #     ])
    #     res = DigitCaps().squash(u)
    #     print(res);

    def test_similarity(self):
        torch.manual_seed(127)

        u = torch.tensor([[[0.5, 1.0],
                        [1.0, 1.0],
                        [1.0, 2.0]]])

        batch_size, in_capsules, in_dimension = u.size()
        out_capsules = 2
        out_dimension = 2

        # from github
        capsule = DigitCaps(out_capsules, in_capsules, in_dimension, out_dimension)

        res_didigtCaps = capsule.forward(u)

        print("res digitcaps", res_didigtCaps)
        # print("res2:", res2)

        # my implementation
        W, res_custom = procedure1(u.squeeze(0), out_capsules, out_dimension, W=capsule.W.squeeze(0))

        self.assertTrue(torch.allclose(res_didigtCaps, res_custom), "Custom implementation returns different result")

    def test_similarity2(self):
        torch.manual_seed(127)

        u = torch.tensor([[[0.5, 1.0, 0.1],
                        [1.0, 1.0, 0.0],
                        [1.0, 2.0, 3.0],
                        [2.0, 3.0, 4.0]]])

        batch_size, in_capsules, in_dimension = u.size()
        out_capsules = 6
        out_dimension = 1

        # from github
        capsule = DigitCaps(out_capsules, in_capsules, in_dimension, out_dimension)

        res_didigtCaps = capsule.forward(u)

        print("res digitcaps", res_didigtCaps)
        # print("res2:", res2)

        # my implementation
        W, res_custom = procedure1(u.squeeze(0), out_capsules, out_dimension, W=capsule.W.squeeze(0))

        self.assertTrue(torch.allclose(res_didigtCaps, res_custom), "Custom implementation returns different result")

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()