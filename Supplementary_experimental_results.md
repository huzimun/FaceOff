We conduct an ablation study on the the scaling factor $\lambda$ to analyze its impact on protection efficacy across four target models. 
To better compare with the individual target loss and deviation loss, we use a ratio $w$ to replace $\lambda$ in the FaceOff loss function:
- FaceOff loss function in our paper:

    $L_{FaceOff} = L_{target} + \lambda * L_{deviation}$

- Replace the scaling factor $\lambda$ with a ratio $w$:

    $L_{FaceOff} = (1-w) * L_{target} + w * L_{deviation}$

The ratio $w$ is set to 0, 0.25, 0.5, 0.75, and 1 respectively and the results are summarized in the following table. We can see that the best performance on IP-Adapter, Fastcomposer and Face-diffuser is achieved when $w=0.5$. For DreamBooth, the best performance is achieved when $w=0.25$.
We can also observe that the IMS values of w= 0.25, 0.5, and 0.75 are always less than the maximum IMS values of w=0 (i.e. target loss) and w=1 (i.e. deviation loss). That is to say, the joint FaceOff loss will not result in a worse situation than the individual target or deviation loss. This observation contributes to more diverse and effective perturbation patterns.

| Model  | IP-Adapter | IP-Adapter | Fastcomposer | Fastcomposer | Face-diffuser | Face-diffuser | DreamBooth | DreamBooth |
|--------|--------------------|--------------------|-----------------------|-----------------------|------------------------|------------------------|----------------------|----------------------|
| Method | $IMS_{ARC}$↓                      | $IMS_{VGG}$↓                      | $IMS_{ARC}$↓                        | $IMS_{VGG}$↓                        | $IMS_{ARC}$↓                        | $IMS_{VGG}$↓                        | $IMS_{ARC}$↓                        | $IMS_{VGG}$↓                        |
| No Def.| 0.37±0.08                    | 0.76±0.07                    | 0.38±0.079                     | 0.77±0.07                      | 0.39±0.08                      | 0.77±0.07                      | 0.57±0.06                      | 0.82±0.06                      |
| 0      | 0.05±0.06                    | 0.35±0.23                    | 0.18±0.09                      | 0.60±0.11                      | 0.18±0.08                      | 0.60±0.11                      | 0.16±0.10                      | 0.51±0.23                      |
| 0.25   | **0.04±0.04**                    | 0.32±0.22                    | 0.17±0.08                      | 0.58±0.12                      | 0.17±0.08                      | 0.59±0.12                      | **0.11±0.09**                      | **0.41±0.22**                      |
| 0.5    | **0.04±0.06**                  | **0.28±0.24**                    | **0.15±0.08**                      | **0.56±0.13**                      | **0.15±0.08**                      | **0.57±0.13**                     | 0.25±0.10                      | 0.64±0.18                      |
| 0.75 | 0.06±0.05                    | 0.43±0.13                    | **0.15±0.09**                      | 0.59±0.12                      | **0.15±0.09**                      | 0.60±0.13                      | 0.38±0.09                      | 0.75±0.10                      |
| 1      | 0.07±0.05           | 0.43±0.10           | 0.17±0.10             | 0.61±0.13             | 0.18±0.10              | 0.61±0.13              | 0.38±0.09            | 0.75±0.08            |