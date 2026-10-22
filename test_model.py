import unittest
import numpy as np
from model import *
from optics import image,fit_center,derivative,SIGMA,FLUX,BACKGROUND
class Tests(unittest.TestCase):
    def test_rotation_preserves_population_spectrum(self):
        for a in (0,30,60,90):np.testing.assert_allclose(np.linalg.eigvalsh(population_lag(a)),RHO,atol=1e-14)
        self.assertGreater(np.linalg.norm(population_lag(60)-population_lag(0)),.9)
    def test_stationary_covariance_and_lag(self):
        z=sequence(np.random.default_rng(17),80000,60)
        np.testing.assert_allclose(z.T@z/len(z)/SCALE**2,np.eye(2),atol=.04)
        np.testing.assert_allclose(lag(z),population_lag(60),atol=.04)
    def test_mean_covariance_against_double_sum(self):
        m=7;q=rotation(31)
        brute=sum((q*RHO**abs(i-j))@q.T for i in range(m) for j in range(m))*SCALE**2/m**2
        np.testing.assert_allclose(mean_covariance(m,31),brute,atol=1e-18)
    def test_mean_variance_monte_carlo(self):
        rng=np.random.default_rng(123)
        z=np.array([sequence(rng,12,60).mean(0) for _ in range(6000)])
        np.testing.assert_allclose(np.cov(z.T),mean_covariance(12,60),atol=5e-6)
    def test_gls_weights_minimize_variance(self):
        v=mean_covariance(32,60);w=weights(32,60)
        self.assertAlmostEqual(w.sum(),1)
        for a in np.linspace(-1,2,201):
            b=np.array([a,1-a]);self.assertLessEqual(w@v@w,b@v@b+1e-15)
    def test_common_basis_rotation_invariance(self):
        rng=np.random.default_rng(9);a=sequence(rng,96,0);b=sequence(rng,96,60);q=rotation(37)
        np.testing.assert_allclose(scores(a@q,b@q),scores(a,b),atol=1e-14)
    def test_rank_threshold(self):
        x=np.column_stack([np.arange(999),np.arange(999)[::-1]])
        np.testing.assert_array_equal(threshold(x),[949,949])
    def test_pixel_flux_and_derivative(self):
        self.assertAlmostEqual((image(.2)-BACKGROUND).sum(),FLUX,places=8)
        np.testing.assert_allclose(derivative(.2),(image(.200001)-image(.199999))/2e-6,rtol=1e-6,atol=1e-6)
    def test_noiseless_optical_time(self):
        for c in (-.6,.23,.326,.8):self.assertAlmostEqual(fit_center(image(c)),c,places=9)
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):lag(np.ones((2,2)))
        with self.assertRaises(ValueError):threshold(np.ones((2,2)))
        with self.assertRaises(ValueError):image(0,sigma=0)
if __name__=='__main__':unittest.main()
