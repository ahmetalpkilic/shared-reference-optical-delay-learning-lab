import unittest
import numpy as np
from whitening import covariance,inverse_root,matrix_score,calibration
from model import sequence,SCALE,scores,rotation

class WhiteningTests(unittest.TestCase):
    def test_inverse_root_identity(self):
        c=np.array([[4.,1.],[1.,2.]])*1e-4;w=inverse_root(c)
        np.testing.assert_allclose(w@c@w.T,np.eye(2),atol=1e-14)
    def test_covariance_hand_calculation(self):
        x=np.array([[1,0],[0,1],[-1,0],[0,-1.]])
        np.testing.assert_allclose(covariance(x),np.eye(2)*2/3)
    def test_known_c_recovers_previous_score(self):
        rng=np.random.default_rng(5);a=sequence(rng,96,0);b=sequence(rng,96,60)
        self.assertAlmostEqual(matrix_score(a,b,np.eye(2)*SCALE**2),scores(a,b)[1])
    def test_common_affine_invariance(self):
        rng=np.random.default_rng(6);a=sequence(rng,96,0);b=sequence(rng,96,60)
        c=calibration(rng,16);t=np.array([[2.,.7],[.1,.8]])
        self.assertAlmostEqual(matrix_score(a,b,c),matrix_score(a@t.T,b@t.T,t@c@t.T),places=12)
    def test_covariance_scale_changes_score(self):
        rng=np.random.default_rng(7);a=sequence(rng,96,0);b=sequence(rng,96,60);c=calibration(rng,16)
        self.assertAlmostEqual(matrix_score(a,b,c)/2,matrix_score(a,b,2*c))
    def test_large_independent_calibration_converges(self):
        c=calibration(np.random.default_rng(8),100000)
        np.testing.assert_allclose(c/SCALE**2,np.eye(2),atol=.012)
    def test_singular_rejected(self):
        with self.assertRaises(ValueError):inverse_root(np.ones((2,2)))
        with self.assertRaises(ValueError):covariance(np.ones((2,2)))
if __name__=='__main__':unittest.main()
