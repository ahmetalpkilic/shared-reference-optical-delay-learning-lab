import unittest
import numpy as np
from lab import reference_differences, sequences, lag_batch, N
from model import SCALE

class ReferenceTests(unittest.TestCase):
    def test_fixed_reference_subtracted_exactly(self):
        ref=np.array([[1.,.2],[.2,3.]])*SCALE**2
        actual=reference_differences(np.random.default_rng(42),5,60,ref)
        expected=lag_batch(sequences(np.random.default_rng(42),5,N,60))-ref
        np.testing.assert_array_equal(actual,expected)

    def test_reference_translation_changes_all_events_equally(self):
        a=reference_differences(np.random.default_rng(43),8,0,np.zeros((2,2)))
        ref=np.eye(2)*SCALE**2
        b=reference_differences(np.random.default_rng(43),8,0,ref)
        np.testing.assert_allclose(a-b,np.broadcast_to(ref,a.shape),atol=1e-18)

    def test_shared_reference_induces_cross_event_covariance(self):
        # Independent scalar analogue: d1=u1-r, d2=u2-r => covariance Var(r).
        rng=np.random.default_rng(44)
        u1,u2=rng.normal(size=(2,100000));r=rng.normal(0,2,100000)
        self.assertAlmostEqual(np.cov(u1-r,u2-r)[0,1],4.,delta=.08)
        fresh=rng.normal(0,2,100000)
        self.assertAlmostEqual(np.cov(u1-r,u2-fresh)[0,1],0.,delta=.08)

    def test_conditional_rank_uniformity_fixed_reference(self):
        # Extreme fixed reference: matching ranks calibrated, fresh-reference
        # training not calibrated conditionally. Independent scalar special case.
        rng=np.random.default_rng(45)
        train=np.abs(rng.normal(size=(3000,99))-4.)
        cuts=np.sort(train,axis=1)[:,94] # ceil(100*.95), one-based
        alarm=np.abs(rng.normal(size=3000)-4.)>cuts
        self.assertAlmostEqual(alarm.mean(),.05,delta=.015)
        wrong=np.quantile(np.abs(rng.normal(size=200000)-rng.normal(size=200000)),.95)
        self.assertGreater(np.mean(np.abs(rng.normal(size=3000)-4.)>wrong),.8)

if __name__=='__main__':unittest.main()
