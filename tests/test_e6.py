import sys, unittest, json, tempfile
from pathlib import Path
import numpy as np, torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from experiment_e6 import slice_trigger,gen_data
from experiment_e5 import Net,AffineFlow,verify_box,logits
from statistics_e5 import sign_p,holm
class ResearchTests(unittest.TestCase):
    def test_fixed_trigger_exact_equivalence(self):
        torch.manual_seed(1);n=Net(3,8).eval();x=torch.randn(20,2)
        for t in (0.,1.):
            with torch.no_grad():
                self.assertLess((n(torch.cat((x,torch.full((20,1),t)),1))-slice_trigger(n,t)(x)).abs().max().item(),1e-6)
    def test_ties_are_not_significant(self):
        self.assertEqual(sign_p([0,0,0]),1.0)
        self.assertEqual(sign_p([0,1,1]),0.5)
    def test_holm_monotonic(self):
        p=holm([0.001,.02,.5]);self.assertTrue(all(0<=v<=1 for v in p));self.assertEqual(p[0],.003)
    def test_solver_witness_vs_samples(self):
        rng=np.random.default_rng(1);flow=AffineFlow(rng.normal(size=(100,2)))
        torch.manual_seed(1);m1=Net(2,5).eval();m2=Net(2,5).eval();cert=verify_box(flow,m1,m2,1.,15)
        self.assertEqual(cert['status'],'optimal')
        sample=flow.forward(rng.uniform(-1,1,size=(200,2)))
        empirical=np.abs(logits(m1,sample)-logits(m2,sample)).max()
        self.assertGreaterEqual(cert['maximum']+1e-4,empirical)
    def test_disjoint_flow_fit_from_canaries(self):
        d=gen_data('digits','./data',2026,24)
        self.assertEqual(len(np.intersect1d(d['canary'],d['retain'])),0)
        self.assertTrue(np.isfinite(d['r']))
        self.assertTrue(0<=d['coverage']<=1)
if __name__=='__main__':unittest.main()
