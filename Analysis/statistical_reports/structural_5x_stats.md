# Structural 5x Stats

## A2C: Inverted vs Normal
- n_inv=30, n_norm=30
- mean_inv=723337.50 ± 1060.57
- mean_norm=661165.41 ± 1721.22
- diff=62172.09 (t=165.604, p=3.874e-68, d=42.759)

## PPO: Inverted vs Normal
- n_inv=30, n_norm=30
- mean_inv=722568.30 ± 354.23
- mean_norm=659197.89 ± 396.87
- diff=63370.41 (t=641.512, p=3.938e-112, d=165.638)

## Combined: Inverted vs Normal (A2C+PPO)
- n_inv=60, n_norm=60
- mean_inv=722952.90 ± 879.24
- mean_norm=660181.65 ± 1589.92
- diff=62771.25 (t=265.381, p=1.701e-134, d=48.452)

## Stability Proxies (means)
shape,algorithm,max_load_mean,drift_l1_mean,lyapunov_mean,crash_mean
inverted_pyramid,A2C,0.862500011920929,0.08147175942028984,3.5291392000000004,0.0
inverted_pyramid,PPO,0.8625067093278408,0.08258633333333333,3.9564697333333334,0.0
normal_pyramid,A2C,0.862500011920929,0.05668855942028985,1.7871572,0.0
normal_pyramid,PPO,0.8626930703819354,0.057156886956521734,1.8257922666666666,0.0
