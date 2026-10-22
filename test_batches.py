import unittest
import numpy as np
from lab import sequences,lag_batch,score,variance_components
from model import sequence,lag,SCALE
from whitening import matrix_score

class BatchTests(unittest.TestCase):
    def test_vectorized_trace_matches_scalar(self):
        for angle in (0,60):
            np.testing.assert_allclose(sequences(np.random.default_rng(3),1,96,angle)[0],sequence(np.random.default_rng(3),96,angle))
    def test_dimensional_lag_and_score(self):
        a,b=sequences(np.random.default_rng(4),2,96,60)
        c=np.array([[2.,.3],[.3,1.]])*SCALE**2
        np.testing.assert_allclose(lag_batch(a[None])[0],lag(a)*SCALE**2,atol=1e-18)
        self.assertAlmostEqual(score(lag_batch(b[None])-lag_batch(a[None]),c)[0],matrix_score(a,b,c))
    def test_fixed_covariance_scale_and_rank(self):
        rng=np.random.default_rng(5);d=rng.normal(size=(999,2,2));c=np.eye(2)*SCALE**2
        a=score(d,c);b=score(d,c*3)
        np.testing.assert_allclose(a/3,b)
        # Matching training absorbs a scalar normalization change exactly.
        np.testing.assert_array_equal(a>np.sort(a)[949],b>np.sort(b)[949])
    def test_variance_decomposition_independent_mixture(self):
        rng=np.random.default_rng(6);p=np.tile([.01,.19],10000)
        rates=rng.binomial(200,p)/200
        out=variance_components(rates,200)
        self.assertAlmostEqual(out['estimated_between_batch_variance'],.09**2,delta=.0003)
    def test_no_clipping_negative_variance(self):
        self.assertLess(variance_components([.05]*10,200)['estimated_between_batch_variance'],0)

if __name__=='__main__':unittest.main()
