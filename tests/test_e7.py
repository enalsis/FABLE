import sys, unittest, tempfile, json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from privacy_shadow import gaussian_llr,nearest_unique
from analyze_e7 import make_report
from statistics_e5 import sign_p,holm

class ExtendedTests(unittest.TestCase):
    def test_llr_in_higher_when_in_distribution(self):
        i=np.array([2,2.2,1.8,2.1]);o=np.array([-2,-2.1,-1.8,-2.2]);
        self.assertGreater(gaussian_llr(2,i,o),gaussian_llr(-2,i,o))
    def test_nearest_unique(self):
        x=np.array([[0.],[1.],[.1],[.2]])
        self.assertEqual(set(nearest_unique(x,[0,1],[2,3])),{2,3})
    def test_sign_zero(self):self.assertEqual(sign_p([0,0,0]),1.)
    def test_holm(self):self.assertEqual(holm([.01,.02])[0],.02)
    def test_report_generation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'request_2026';p.mkdir()
            rows=[]
            for m,g in [('original',2.),('reset_head',1.),('retrain_0',0.)]:
                rows.append(dict(seed=11,method=m,test_acc=.9,canary_wrong_label_fit=.3,
                                 canary_wrong_label_margin=.1,probe_trigger_wrong_label_rate=.2,
                                 gap_trigger0=g/2,gap_trigger1=g,status_trigger0='optimal',status_trigger1='optimal'))
            (p/'results.json').write_text(json.dumps({'rows':rows,'comparisons':[]}))
            rep=make_report(d);self.assertEqual(rep['n_requests'],1)
            self.assertTrue((Path(d)/'REPORT.md').exists())
if __name__=='__main__':unittest.main()

class CouplingTests(unittest.TestCase):
    def test_piecewise_milp_matches_dense_grid(self):
        import torch
        from coupling_flow import TriangularCouplingFlow,verify_coupling
        from experiment_e5 import Net,logits
        rng=np.random.default_rng(3);flow=TriangularCouplingFlow(rng.normal(size=(130,2)))
        z=rng.uniform(-1,1,size=(130,2));x=flow.forward(z)
        self.assertLess(np.abs(flow.inverse(x)-z).max(),1e-7)
        torch.manual_seed(4);a=Net(2,5).eval();b=Net(2,5).eval()
        cert=verify_coupling(flow,a,b,1.,15)
        self.assertEqual(cert['status'],'optimal')
        self.assertGreaterEqual(cert['maximum']+1e-4,float(np.abs(logits(a,x)-logits(b,x)).max()))
