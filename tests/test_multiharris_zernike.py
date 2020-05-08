#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 22:52:23 2019

@author: vik748
"""
import sys, os
if os.path.dirname(os.path.realpath(__file__)) == os.getcwd():
    sys.path.insert(1, os.path.join(sys.path[0], '..'))
import cv2
import data
from zernike.zernike import MultiHarrisZernike
import unittest
import numpy as np


class TestMultiHarrisZernike(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        zernike_detector = MultiHarrisZernike(Nfeats= 600, seci = 5, secj = 4, levels = 6,
                                              ratio = 0.75, sigi = 2.75, sigd = 1.0, nmax = 8,
                                              like_matlab=False, lmax_nd = 3, harris_threshold = None)

        data_path = os.path.dirname(os.path.relpath(data.__file__))
        img = cv2.imread(os.path.join(data_path,'contrast_test_images/G0029482.png'),
                         cv2.IMREAD_COLOR)

        assert img is not None, "Couldn't read image"
        gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cls.kp, cls.des = zernike_detector.detectAndCompute(gr, mask=None, timing=False)
        cls.kp_sorted = sorted(cls.kp, key=lambda x: x.response, reverse=True)

    def setup(self):
        self.kp = cls.kp
        self.des = cls.des
        self.kp_sorted = cls.kp_sorted

    def test_zernike_detector_length(self):
        """
        Test the zernike detector number of features
        """

        self.assertEqual(len(self.kp), 640,"Incorrect number of features detected")

    def test_zernike_detector_response(self):
        """
        Test the zernike detector max feature response
        """

        self.assertTrue(abs(self.kp_sorted[0].response - 392.973) < 0.01, "Incorrect max response")

    def test_zernike_descriptor(self):
        """
        Test the zernike descriptor sum
        """
        self.assertTrue(abs(np.sum(self.des[0,:]) - 51.062) < 0.01, "Incorrect sum of feature 0 descriptor")

if __name__ == '__main__':
    unittest.main()