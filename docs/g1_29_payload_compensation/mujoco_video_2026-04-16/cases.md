# MuJoCo Payload Compensation Video Cases

- side: `right`
- payload_mass: `0.8` kg
- payload_com_ee: `[0.02, 0.0, 0.08]` m
- payload_scale: `1.0`
- arm_tau_limit: `None`
- seed: `42`
- num_samples: `500`
- top_k: `3`
- video: `payload_compensation_mujoco.mp4`

| Case | EE Pos Before (mm) | EE Pos After (mm) | EE Ori Before (deg) | EE Ori After (deg) | Joint RMS Before (deg) | Joint RMS After (deg) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 76.295 | 0.000 | 23.109 | 0.000000 | 5.301 | 0.000000 |
| 2 | 73.628 | 0.000 | 22.372 | 0.000000 | 5.190 | 0.000000 |
| 3 | 72.604 | 0.000 | 22.202 | 0.000000 | 5.169 | 0.000000 |