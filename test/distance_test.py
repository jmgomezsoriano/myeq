import unittest

from myeq.distance import inv_norm_sigmoid


class DistanceCase(unittest.TestCase):
    def test_inv_norm_sigmoid(self):
        self.assertEqual(inv_norm_sigmoid(0), 0.9999853027487737)
        self.assertEqual(inv_norm_sigmoid(0, 0.2), 0.9999999399346944)
        self.assertEqual(inv_norm_sigmoid(0, 0.2, 0.8), 0.9999999453951767)
        self.assertEqual(inv_norm_sigmoid(0, 0.2, 0.8, 1.0), 0.9946457192605721)
        self.assertEqual(inv_norm_sigmoid(1), 0.9995881980814431)
        self.assertEqual(inv_norm_sigmoid(1, 0.2), 0.9999910856079368)
        self.assertEqual(inv_norm_sigmoid(1, 0.2, 0.8), 0.9999918960072153)
        self.assertEqual(inv_norm_sigmoid(1, 0.2, 0.8, 1.0), 0.6)
        self.assertEqual(inv_norm_sigmoid(2), 0.9886007197729876)
        self.assertEqual(inv_norm_sigmoid(2, 0.2), 0.9986789596140715)
        self.assertEqual(inv_norm_sigmoid(2, 0.2, 0.8), 0.9987990541946105)
        self.assertEqual(inv_norm_sigmoid(2, 0.2, 0.8, 1.0), 0.20535428073942774)
        self.assertEqual(inv_norm_sigmoid(3), 0.7633315491944042)
        self.assertEqual(inv_norm_sigmoid(3, 0.2), 0.8394655390504062)
        self.assertEqual(inv_norm_sigmoid(3, 0.2, 0.8), 0.8540595809549147)
        self.assertEqual(inv_norm_sigmoid(3, 0.2, 0.8, 1.0), 0.20003631829496193)
        self.assertEqual(inv_norm_sigmoid(5), 0.12303375714537601)
        self.assertEqual(inv_norm_sigmoid(5, 0.1), 0.12000003643145052)
        self.assertEqual(inv_norm_sigmoid(5, 0.2, 0.8), 0.20016274158244407)
        self.assertEqual(inv_norm_sigmoid(5, 0.1, 0.8), 0.20000003311950043)
        self.assertEqual(inv_norm_sigmoid(5, 0.1, 0.8, 1.0), 0.19999999999999996)
        self.assertEqual(inv_norm_sigmoid(100), 0.12)
        self.assertEqual(inv_norm_sigmoid(100, 0.1), 0.12)
        self.assertEqual(inv_norm_sigmoid(100, 0.2, 0.8), 0.19999999999999996)
        self.assertEqual(inv_norm_sigmoid(0, adjust=True), 1.0)


if __name__ == '__main__':
    unittest.main()
