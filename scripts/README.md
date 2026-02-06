## Universal Controllers

To evaluate a universal controller for multiple random morphologies -

```bash
python scripts/analysis/evaluate_univctrl_pretrained.py --task ...
```

To evaluate a universal controller for a specific morphology -

```bash
python scripts/rsl_rl/play_universal.py --task "" --cmdir "" ...
```

The Isc-BALLU-hetero-general task is different from Isc-Vel-BALLU-1-obstacle task just because the observation space consists of the morphology vector.

The task Isc-BALLU-hetero-pretrain is only meant for pretraining the universal controller. If you want to evaluate the universal controller for a specific morphology, you should use the Isc-BALLU-hetero-general task via the @play_universal.py script.

The universal controller is trained by dropping the robot from 1m height. The policy does not generalize if the robot is dropped from a different height.