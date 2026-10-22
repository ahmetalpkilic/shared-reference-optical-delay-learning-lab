"""Two-channel timing modes in a fixed physical coordinate system."""
import numpy as np
RHO = np.array([0., .85])
SCALE = .02  # seconds; marginal covariance is SCALE**2 I

def rotation(degrees):
    a=np.deg2rad(degrees)
    return np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])

def sequence(rng,n,angle):
    """Stationary AR modes; rotate modes, not the physical timestamp labels."""
    if n<3: raise ValueError('Need at least three observations')
    x=rng.normal(size=(n,2))
    for k in range(1,n): x[k]=RHO*x[k-1]+np.sqrt(1-RHO**2)*x[k]
    return SCALE*x@rotation(angle).T

def lag(z):
    """Known-C whitening; shared basis is essential when subtracting matrices."""
    z=np.asarray(z,float)
    if z.ndim!=2 or z.shape[1]!=2 or len(z)<3 or not np.isfinite(z).all():
        raise ValueError('Require finite (n>=3,2) samples')
    x=(z-z.mean(0))/SCALE
    g=x[:-1].T@x[1:]/len(x)
    return (g+g.T)/2

def scores(old,pilot):
    a,b=lag(old),lag(pilot)
    return np.array([np.max(abs(np.linalg.eigvalsh(a)-np.linalg.eigvalsh(b))),
                     np.linalg.norm(a-b,'fro')])

def population_lag(angle):
    q=rotation(angle)
    return (q*RHO)@q.T

def mean_covariance(m,angle):
    # Double sum of Gamma_|i-j| divided by m^2; finite-m, not long-run limit.
    h=np.arange(1,m)
    v=(m+2*np.sum((m-h[:,None])*RHO**h[:,None],axis=0))/m**2
    q=rotation(angle)
    return SCALE**2*(q*v)@q.T

def weights(m,angle):
    a=np.linalg.solve(mean_covariance(m,angle),np.ones(2))
    return a/a.sum()  # unit sum preserves the common timestamp mean

def threshold(training,alpha=.05):
    x=np.asarray(training)
    k=int(np.ceil((len(x)+1)*(1-alpha)))
    if x.ndim!=2 or k>len(x) or k<1:raise ValueError('Invalid rank calibration')
    return np.sort(x,axis=0)[k-1]
