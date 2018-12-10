import torch as th

from unittest import TestCase
from pro_gan_pytorch import CustomLayers as cL

device = th.device("cuda" if th.cuda.is_available() else "cpu")


class Test_equalized_conv2d(TestCase):

    def setUp(self):
        self.conv_block = cL._equalized_conv2d(21, 3, k_size=(3, 3), pad=1)

        # print the Equalized conv block
        print("\nEqualized conv block:\n%s" % str(self.conv_block))

    def test_forward(self):
        mock_in = th.randn(32, 21, 16, 16).to(device)
        mock_out = self.conv_block(mock_in)

        # check output
        self.assertEqual(mock_out.shape, (32, 3, 16, 16))
        self.assertEqual(th.isnan(mock_out).sum().item(), 0)
        self.assertEqual(th.isinf(mock_out).sum().item(), 0)

        # check the weight's scale
        self.assertAlmostEqual(self.conv_block.weight.data.std(), 1, delta=1e-1)

    def tearDown(self):
        # delete the computational resources
        del self.conv_block


class Test_equalized_deconv2d(TestCase):

    def setUp(self):
        self.deconv_block = cL._equalized_deconv2d(21, 3, k_size=(3, 3), pad=1)

        # print the Equalized conv block
        print("\nEqualized conv block:\n%s" % str(self.deconv_block))

    def test_forward(self):
        mock_in = th.randn(32, 21, 16, 16).to(device)
        mock_out = self.deconv_block(mock_in)

        # check output
        self.assertEqual(mock_out.shape, (32, 3, 16, 16))
        self.assertEqual(th.isnan(mock_out).sum().item(), 0)
        self.assertEqual(th.isinf(mock_out).sum().item(), 0)

        # check the weight's scale
        self.assertAlmostEqual(self.deconv_block.weight.data.std(), 1, delta=1e-1)

    def tearDown(self):
        # delete the computational resources
        del self.deconv_block


class Test_equalized_linear(TestCase):

    def setUp(self):
        self.lin_block = cL._equalized_linear(13, 52)

        # print the Equalized conv block
        print("\nEqualized linear block:\n%s" % str(self.lin_block))

    def test_forward(self):
        # test the forward for the first res block
        mock_in = th.randn(32, 13).to(device)
        mock_out = self.lin_block(mock_in)

        # check output
        self.assertEqual(mock_out.shape, (32, 52))
        self.assertEqual(th.isnan(mock_out).sum().item(), 0)
        self.assertEqual(th.isinf(mock_out).sum().item(), 0)

        # check the weight's scale
        self.assertAlmostEqual(self.lin_block.weight.data.std(), 1, delta=1e-1)

    def tearDown(self):
        # delete the computational resources
        del self.lin_block


class Test_PixelwiseNorm(TestCase):

    def setUp(self):
        self.normalizer = cL.PixelwiseNorm()

    def test_forward(self):
        mock_in = th.randn(1, 13, 1, 1).to(device)
        mock_out = self.normalizer(mock_in)

        # check output
        self.assertEqual(mock_out.shape, mock_in.shape)
        self.assertEqual(th.isnan(mock_out).sum().item(), 0)
        self.assertEqual(th.isinf(mock_out).sum().item(), 0)

        # we cannot comment that the norm of the output tensor
        # will always be less than the norm of the input tensor
        # so no more checking can be done

    def tearDown(self):
        # delete the computational resources
        del self.normalizer


class Test_MinibatchStdDev(TestCase):

    def setUp(self):
        self.minStdD = cL.MinibatchStdDev()

    def test_forward(self):
        mock_in = th.randn(1, 13, 16, 16).to(device)
        mock_out = self.minStdD(mock_in)

        # check output
        self.assertEqual(mock_out.shape[1], mock_in.shape[1] + 1)
        self.assertEqual(th.isnan(mock_out).sum().item(), 0)
        self.assertEqual(th.isinf(mock_out).sum().item(), 0)

    def tearDown(self):
        # delete the computational resources
        del self.minStdD
