"""Spectral (FFT power spectrum) + autocorrelation analysis of PM2.5 per station.

Checks whether the daily PM2.5 series carry periodic structure — evidence for
"the data is periodic", which underpins why a frequency-domain model helps.
Reads ../Finalized_Dataset_common/*_merged.csv (Date + covariates + pm25).
"""
import glob, os
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "Finalized_Dataset_common")

def longest_run(mask):
    best_s=best_e=cur_s=0
    for i,v in enumerate(list(mask)+[False]):
        if v and i>0 and mask[i-1]:
            continue
        if v:
            cur_s=i
        else:
            if i>0 and mask[i-1] and (i-cur_s)>(best_e-best_s):
                best_s,best_e=cur_s,i
    return best_s,best_e

def analyze(path):
    df = pd.read_csv(path)
    df['Date']=pd.to_datetime(df['Date'])
    df=df.sort_values('Date').drop_duplicates('Date')
    # daily reindex + short-gap interpolation (<=3d), like the training pipeline
    full=pd.date_range(df['Date'].min(),df['Date'].max(),freq='D')
    s=df.set_index('Date')['pm25'].reindex(full).interpolate(limit=3)
    vals=s.values
    mask=~np.isnan(vals)
    a,b=longest_run(mask)
    seg=vals[a:b]
    if len(seg)<60:
        return None
    seg=seg-seg.mean()
    # FFT power spectrum
    n=len(seg)
    fft=np.fft.rfft(seg*np.hanning(n))
    power=np.abs(fft)**2
    freqs=np.fft.rfftfreq(n,d=1.0)  # cycles/day
    # ignore the lowest freqs (period > n/2, i.e. trend); look at periods 2..90 days
    periods=np.where(freqs>0,1.0/np.maximum(freqs,1e-9),np.inf)
    band=(periods>=2)&(periods<=90)
    top_idx=np.argsort(power[band])[::-1][:5]
    top_periods=periods[band][top_idx]
    # ACF at key lags
    def acf(x,lag):
        x=x-x.mean()
        return np.sum(x[:-lag]*x[lag:])/np.sum(x*x)
    acfs={L:acf(seg,L) for L in [1,7,14,30]}
    return dict(name=os.path.basename(path).replace('_merged.csv',''),
                seg_len=len(seg), top_periods=top_periods, acf=acfs)

rows=[]
for f in sorted(glob.glob(os.path.join(DATA,"*_merged.csv"))):
    r=analyze(f)
    if r: rows.append(r)

print(f"{'station':22s} {'segLen':>6s} {'ACF1':>6s} {'ACF7':>6s} {'ACF14':>6s} {'ACF30':>6s}  top periods(days)")
for r in rows:
    tp=", ".join(f"{p:.1f}" for p in r['top_periods'])
    print(f"{r['name']:22s} {r['seg_len']:6d} {r['acf'][1]:6.2f} {r['acf'][7]:6.2f} "
          f"{r['acf'][14]:6.2f} {r['acf'][30]:6.2f}  [{tp}]")

# aggregate: dominant period across stations
allp=np.concatenate([r['top_periods'] for r in rows])
print(f"\nMost common dominant periods (rounded): ")
vc=pd.Series(np.round(allp).astype(int)).value_counts().head(8)
for p,c in vc.items():
    print(f"  ~{p} days: appears in {c} station-peaks")
print(f"\nMean ACF: lag1={np.mean([r['acf'][1] for r in rows]):.2f}  "
      f"lag7={np.mean([r['acf'][7] for r in rows]):.2f}  "
      f"lag14={np.mean([r['acf'][14] for r in rows]):.2f}  "
      f"lag30={np.mean([r['acf'][30] for r in rows]):.2f}")
