"""Shared reference within fixed-calibration alarm batches; all times in seconds."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import SCALE, RHO, rotation, threshold, weights, mean_covariance
from whitening import calibration, inverse_root
from optics import image, fit_center, ORIGIN, VELOCITY, READ_SD

SEED, TRAIN, BATCHES, EVENTS, N, M = 20261022, 999, 32, 200, 96, 32
SIZES, TAU = (8, 128), .12


def sequences(rng, count, n, angle):
    """Vectorized independent stationary AR traces, shape (count,n,2)."""
    x = rng.normal(size=(count,n,2))
    for t in range(1,n):
        x[:,t] = RHO*x[:,t-1]+np.sqrt(1-RHO**2)*x[:,t]
    return SCALE*x@rotation(angle).T


def lag_batch(x):
    z = x-x.mean(axis=1,keepdims=True)
    g = np.einsum('bni,bnj->bij',z[:,:-1],z[:,1:])/x.shape[1]
    return (g+g.transpose(0,2,1))/2


def differences(rng,count,angle):
    return lag_batch(sequences(rng,count,N,angle))-lag_batch(sequences(rng,count,N,0))


def reference_differences(rng,count,angle,reference):
    """Subtract the SAME dimensional lag matrix; no old-reference redraw."""
    return lag_batch(sequences(rng,count,N,angle))-reference


def score(delta,c):
    w=inverse_root(c)
    return np.linalg.norm(w@delta@w.T,axis=(-2,-1))


def variance_components(rates,events):
    """Unbiased method-of-moments estimate; negative estimates retained."""
    rates=np.asarray(rates)
    within=np.mean(rates*(1-rates)/(events-1))
    return dict(observed_rate_variance=float(rates.var(ddof=1)),
                estimated_binomial_variance=float(within),
                estimated_between_batch_variance=float(rates.var(ddof=1)-within))


def run():
    train_pair,batch_cov,conditional_pair,reference_rng,test_pair,clock_rng,camera_rng=[
        np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(7)]
    known=np.eye(2)*SCALE**2
    d=differences(train_pair,TRAIN,0)
    cuts={'known':float(threshold(score(d,known)[:,None])[0])}
    w0,w1=weights(M,0),weights(M,60)
    rows=[]
    for b in range(BATCHES):
        covs={k:calibration(batch_cov,k) for k in SIZES}
        # One noisy old reference is frozen for the WHOLE deployment batch.
        reference=lag_batch(sequences(reference_rng,1,N,0))[0]
        # Covariance-only training still averages over fresh old references.
        train=differences(conditional_pair,TRAIN,0)
        local={k:float(threshold(score(train,c)[:,None])[0]) for k,c in covs.items()}
        # Full conditioning uses this deployed reference, but fresh null pilots.
        fixed_train=reference_differences(conditional_pair,TRAIN,0,reference)
        full={k:float(threshold(score(fixed_train,c)[:,None])[0]) for k,c in covs.items()}
        for angle in (0,60):
            delta=reference_differences(test_pair,EVENTS,angle,reference)
            means=sequences(clock_rng,EVENTS,M,angle).mean(axis=1)
            camera=image(ORIGIN+VELOCITY*TAU)+camera_rng.normal(0,READ_SD,(EVENTS,33))
            optical=np.array([(fit_center(y)-ORIGIN)/VELOCITY-TAU for y in camera])
            error=np.column_stack((optical-means@w0,optical-means@w1))*1000
            policies=[('known',0,known,cuts['known'])]
            for k,c in covs.items():
                policies += [('cov-only',k,c,local[k]),('cov+ref',k,c,full[k])]
            for policy,k,c,cut in policies:
                alarm=score(delta,c)>cut
                e=np.where(alarm,error[:,1],error[:,0])
                selected=np.where(alarm[:,None],w1,w0)
                rows.append(dict(batch=b,angle_deg=angle,policy=policy,k=k,
                    covariance_s2=c.tolist(),reference_lag_s2=reference.tolist(),threshold=cut,alarm_rate=float(alarm.mean()),
                    mse_ms2=float(np.mean(e**2)),
                    clock_mse_ms2=float(1e6*np.einsum('ni,ij,nj->n',selected,mean_covariance(M,angle),selected).mean())))
    summaries=[]
    for angle in (0,60):
        for policy,k in [('known',0)]+[(p,k) for k in SIZES for p in ('cov-only','cov+ref')]:
            subset=[r for r in rows if r['angle_deg']==angle and r['policy']==policy and r['k']==k]
            rates=np.array([r['alarm_rate'] for r in subset])
            mses=np.array([r['mse_ms2'] for r in subset])
            summaries.append(dict(angle_deg=angle,policy=policy,k=k,
                mean_alarm_percent=float(100*rates.mean()),
                batch_alarm_percent_q10_q90=(100*np.quantile(rates,[.1,.9])).tolist(),
                alarm_cluster_mcse_percent=float(100*rates.std(ddof=1)/np.sqrt(BATCHES)),
                pooled_delay_rmse_ms=float(np.sqrt(mses.mean())),
                batch_rmse_ms_q10_q90=np.quantile(np.sqrt(mses),[.1,.9]).tolist(),
                mean_clock_mse_ms2=float(np.mean([r['clock_mse_ms2'] for r in subset])),
                **variance_components(rates,EVENTS)))
    return dict(seed=SEED,batches=BATCHES,events_per_batch=EVENTS,training_pairs=TRAIN,
                pilot_length=N,final_repeats=M,known_covariance_marginal_threshold=cuts,summary=summaries,batches_detail=rows)


def plot(result,path):
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    styles=[('known',0),('cov-only',8),('cov+ref',8),('cov-only',128),('cov+ref',128)]
    for ax,angle,key,title in [(axes[0],0,'alarm_rate','Null alarm rate (%)'),
                              (axes[1],60,'alarm_rate','60° detection (%)'),
                              (axes[2],60,'mse_ms2','60° delay RMSE (ms)')]:
        for j,(policy,k) in enumerate(styles):
            v=np.array([r[key] for r in result['batches_detail'] if r['angle_deg']==angle and r['policy']==policy and r['k']==k])
            v=np.sqrt(v) if key=='mse_ms2' else v*100
            ax.scatter(np.full(len(v),j),v,s=13,alpha=.55)
            ax.plot([j-.22,j+.22],[v.mean()]*2,color='black')
        ax.set_xticks(range(5),['known C','cov only\nK=8','cov + ref\nK=8','cov only\nK=128','cov + ref\nK=128'],rotation=25,ha='right',fontsize=8)
        ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].axhline(5,color='gray',ls=':')
    fig.suptitle('Each dot: one covariance + shared reference; 200 fresh pilots')
    fig.savefig(path,dpi=160);plt.close(fig)


if __name__=='__main__':
    out=Path('examples');out.mkdir(exist_ok=True)
    result=run();(out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    plot(result,out/'batches.png')
    for r in result['summary']:
        print(f"angle={r['angle_deg']:2} {r['policy']:11} K={r['k']:3} alarm={r['mean_alarm_percent']:5.2f}% batch10–90={r['batch_alarm_percent_q10_q90']} RMSE={r['pooled_delay_rmse_ms']:.3f} ms")
