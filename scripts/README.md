## Universal Controllers

To train a universal controller on multiple morphologies -

```bash
python scripts/rsl_rl/train.py --task Isc-BALLU-hetero-pretrain-moe --num_envs 1024 --max_iterations 3000 --headless --GCR_range 0.75 0.89 --spcf_range 0.001 0.01 --run_name hard_k1n5_moe_univ_controller_div_seed0 --seed 0
```

To evaluate a universal controller for multiple random morphologies -

```bash
python scripts/analysis/evaluate_univctrl_pretrained.py --task Isc-BALLU-hetero-pretrain-moe --load_run 2026-02-03_04-45-18_hard_k1n5_moe_univ_controller_div_seed0 --checkpoint model_best.pt --difficulty_level 20 --trials 17
```

To evaluate a universal controller for a specific morphology -

```bash
env BALLU_USD_REL_PATH=morphologies/hetero_library_hvyBloon_lab01.20.26/hetero_0002_fl0.480_tl0.443/hetero_0002_fl0.480_tl0.443.usd python scripts/rsl_rl/play_universal.py --task Isc-BALLU-hetero-general-moe --load_run 2026-02-03_04-45-18_hard_k1n5_moe_univ_controller_div_seed0 --checkpoint model_best.pt --num_envs 1 --video --video_length 399 --headless --difficulty_level 19 --GCR 0.83 --spcf=0.005 --cmdir tests
```

The Isc-BALLU-hetero-general task is different from Isc-Vel-BALLU-1-obstacle task just because the observation space consists of the morphology vector.

The task Isc-BALLU-hetero-pretrain is only meant for pretraining the universal controller. If you want to evaluate the universal controller for a specific morphology, you should use the Isc-BALLU-hetero-general task via the @play_universal.py script.

The universal controller is trained by dropping the robot from 1m height. The policy does not generalize if the robot is dropped from a different height.