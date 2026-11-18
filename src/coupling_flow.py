"""Expressive piecewise-affine flow and exact MILP verification by partitioning.

Only a 2D triangular additive coupling is implemented. The 2 branches (z0>=0
and z0<=0) are affine with a global unit determinant relative to a Gaussian
base matrix, so both branches are exactly representable by a verifier.
"""
import numpy as np
from scipy.optimize import minimize
from experiment_e5 import AffineFlow,verify_box

class _Piece:
    def __init__(self,mu,L):self.mu=np.asarray(mu);self.L=np.asarray(L);self.d=2
    def forward(self,z):return np.asarray(z)@self.L.T+self.mu

class TriangularCouplingFlow:
    def __init__(self,examples,shrink=.15,iterations=100):
        raw=np.asarray(examples,dtype=float)
        if raw.ndim!=2 or raw.shape[1]!=2:raise ValueError('2D only')
        self.base=AffineFlow(raw,shrink=shrink)
        z=self.base.inverse(raw)
        # Triangular additive coupling: y1=z1, y2=z2+a relu(z1)+b relu(-z1).
        # MLE estimates coupling coefficients in Gaussian base by maximum
        # likelihood, with modest ridge regularization for small samples.
        v1=np.maximum(z[:,0],0);v2=np.maximum(-z[:,0],0)
        # Inverse: u2 = z2 - a relu(z1) - b relu(-z1)
        feats=np.column_stack([v1,v2]);target=z[:,1]
        coef=np.linalg.solve(feats.T@feats+.2*np.eye(2),feats.T@target)
        self.a,self.b=map(float,coef)
        self.d=2;self.mu=self.base.mu;self.L=self.base.L
    def inverse(self,x):
        z=self.base.inverse(x);z=np.asarray(z,dtype=float)
        z[...,1]-=self.a*np.maximum(z[...,0],0)+self.b*np.maximum(-z[...,0],0)
        return z
    def forward(self,u):
        z=np.asarray(u,dtype=float).copy()
        z[...,1]+=self.a*np.maximum(z[...,0],0)+self.b*np.maximum(-z[...,0],0)
        return self.base.forward(z)
    def radius(self,x,q):
        import math
        score=np.max(abs(self.inverse(x)),axis=1);k=math.ceil((len(score)+1)*q)
        return float(np.sort(score)[k-1]) if k<=len(score) else float('inf')
    def pieces(self,r):
        # When first latent coordinate is nonnegative, transform:
        # z2 <- u2 + a*u1. When negative, z2 <- u2 - b*u1.
        result=[]
        for sign,coefficient in [('pos',self.a),('neg',-self.b)]:
            A=np.array([[1.,0.],[coefficient,1.]])
            L=self.base.L@A
            lo=np.array([0.,-r]) if sign=='pos' else np.array([-r,-r])
            hi=np.array([r,r]) if sign=='pos' else np.array([0.,r])
            result.append((_Piece(self.base.mu,L),(lo,hi),sign))
        return result

def verify_coupling(flow,a,b,r,timeout):
    if not np.isfinite(r):return {'status':'not_certified','reason':'infinite radius'}
    certs=[]
    for piece,bounds,side in flow.pieces(r):
        c=verify_box(piece,a,b,r,timeout,latent_bounds=bounds);c['piece']=side;certs.append(c)
    if any(c['status']!='optimal' for c in certs):
        return {'status':'not_certified','branches':certs,'seconds':sum(c['seconds'] for c in certs)}
    selected=max(certs,key=lambda c:c['maximum'])
    z=np.asarray(selected['z']);reproduced=flow.forward(z)
    # The piece.forward and the global nonlinear forward must agree.
    if not np.allclose(reproduced,selected['witness'],atol=1e-5):
        return {'status':'not_certified','reason':'coupling witness mismatch','branches':certs}
    return {'status':'optimal','maximum':selected['maximum'],'witness':selected['witness'],
            'z':selected['z'],'reproduced':selected['reproduced'],
            'seconds':sum(c['seconds'] for c in certs),'branch_optima':[{'side':c['piece'],'maximum':c['maximum']} for c in certs]}

def search_coupling(flow,a,b,r,n_random=1000,pgd_restarts=8,pgd_steps=50,seed=0):
    import torch
    from experiment_e5 import logits
    rng=np.random.default_rng(seed)
    u=rng.uniform(-r,r,size=(n_random,2));x=flow.forward(u)
    randommax=float(np.max(abs(logits(a,x)-logits(b,x))))
    z=torch.as_tensor(rng.uniform(-r,r,size=(pgd_restarts,2)),dtype=torch.float32).requires_grad_()
    L=torch.as_tensor(flow.base.L,dtype=torch.float32);mu=torch.as_tensor(flow.base.mu,dtype=torch.float32)
    best=0.
    for _ in range(pgd_steps):
        diff=z[:,0:1];nonlin=flow.a*torch.relu(diff)+flow.b*torch.relu(-diff)
        v=torch.cat([diff,z[:,1:2]+nonlin],dim=1)
        pred=a(v@L.T+mu)-b(v@L.T+mu)
        best=max(best,float(pred.detach().abs().max()))
        grad=torch.autograd.grad(pred.abs().sum(),z)[0]
        with torch.no_grad():z.add_(.08*r*grad.sign()).clamp_(-r,r)
    with torch.no_grad():
        diff=z[:,0:1];v=torch.cat([diff,z[:,1:2]+flow.a*torch.relu(diff)+flow.b*torch.relu(-diff)],dim=1)
        best=max(best,float((a(v@L.T+mu)-b(v@L.T+mu)).abs().max()))
    return {'random_max':randommax,'pgd_max':best,'n_random':n_random,'pgd_restarts':pgd_restarts,'pgd_steps':pgd_steps}
