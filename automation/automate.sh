# Exp 1: Reward in Base Frame (redoing after actuation refactor)
python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-real-priv --num_envs 16 \
                                --run_name "real_bf" --max_iterations 10_000 \
                                env.rewards.track_lin_vel_xy_world_exp.weight=0.0 \
                                --headless

# Exp 2: Reward in World Frame (redoing after actuation refactor)
python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-real-priv --num_envs 16 \
                                --run_name "real_wf" --max_iterations 10_000 \
                                env.rewards.track_lin_vel_xy_base_exp.weight=0.0 \
                                --headless

# Exp 3: Scaling Exp 1 to 512 envs
python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-real-priv --num_envs 512 \
                                --run_name "real_bf_envs_512" --max_iterations 10_000 \
                                env.rewards.track_lin_vel_xy_world_exp.weight=0.0 \
                                --headless

# Exp 4: Use stricter reward in base frame
python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-real-priv --num_envs 16 \
                                --run_name "real_bf_std=0.2" --max_iterations 10_000 \
                                env.rewards.track_lin_vel_xy_world_exp.weight=0.0 \
                                env.rewards.track_lin_vel_xy_base_exp.params.std=0.2 \
                                --headless

# Exp 5: Use stricter reward in world frame
python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-real-priv --num_envs 16 \
                                --run_name "real_wf_std=0.2" --max_iterations 10_000 \
                                env.rewards.track_lin_vel_xy_base_exp.weight=0.0 \
                                env.rewards.track_lin_vel_xy_world_exp.params.std=0.2 \
                                --headless

# Exp 6: Repeat Exp1 with spring_coeff=0.001
python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-real-priv --num_envs 16 \
                                --run_name "real_bf_spr_coef=0.001" --max_iterations 8_000 \
                                env.rewards.track_lin_vel_xy_world_exp.weight=0.0 \
                                env.scene.robot.actuators.knee_effort_actuators.spring_coeff=0.001 \
                                --headless

# Exp 7: Repeat Exp1 with spring_coeff=0.01
python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-real-priv --num_envs 16 \
                                --run_name "real_bf_spr_coef=0.01" --max_iterations 8_000 \
                                env.rewards.track_lin_vel_xy_world_exp.weight=0.0 \
                                env.scene.robot.actuators.knee_effort_actuators.spring_coeff=0.01 \
                                --headless

# Exp 8: Repeat Exp1 with spring_coeff=0.1
python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-real-priv --num_envs 16 \
                                --run_name "real_bf_spr_coef=0.1" --max_iterations 8_000 \
                                env.rewards.track_lin_vel_xy_world_exp.weight=0.0 \
                                env.scene.robot.actuators.knee_effort_actuators.spring_coeff=0.1 \
                                --headless

# Exp 9: Repeat Exp6 in world frame
python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-real-priv --num_envs 16 \
                                --run_name "real_wf_spr_coef=0.001" --max_iterations 8_000 \
                                env.rewards.track_lin_vel_xy_base_exp.weight=0.0 \
                                env.scene.robot.actuators.knee_effort_actuators.spring_coeff=0.001 \
                                --headless

# Exp 10: Repeat Exp7 in world frame
python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-real-priv --num_envs 16 \
                                --run_name "real_wf_spr_coef=0.01" --max_iterations 8_000 \
                                env.rewards.track_lin_vel_xy_base_exp.weight=0.0 \
                                env.scene.robot.actuators.knee_effort_actuators.spring_coeff=0.01 \
                                --headless

# Exp 11: Repeat Exp8 in world frame
python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-real-priv --num_envs 16 \
                                --run_name "real_wf_spr_coef=0.1" --max_iterations 8_000 \
                                env.rewards.track_lin_vel_xy_base_exp.weight=0.0 \
                                env.scene.robot.actuators.knee_effort_actuators.spring_coeff=0.1 \
                                --headless


#python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-mocap --num_envs 16 \
#                                --run_name "mocap" --max_iterations 700 \
#                                --headless agent.policy.init_noise_std=1.0

# python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-mocap --num_envs 16 \
#                                 --run_name "mocap" --max_iterations 700 \
#                                 --headless agent.policy.init_noise_std=0.1

# python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-mocap --num_envs 16 \
#                                 --run_name "mocap" --max_iterations 700 \
#                                 --headless agent.policy.init_noise_std=2.0

# python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-mocap --num_envs 16 \
#                                 --run_name "mocap" --max_iterations 700 \
#                                 --headless agent.policy.actor_hidden_dims=[64, 32, 16] \
#                                 agent.policy.critic_hidden_dims=[64, 32, 16]

# python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-mocap --num_envs 16 \
#                                 --run_name "mocap" --max_iterations 700 \
#                                 --headless agent.algorithm.value_loss_coeff=0.5 

# python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-mocap --num_envs 16 \
#                                 --run_name "mocap" --max_iterations 700 \
#                                 --headless agent.algorithm.value_loss_coeff=10.0

# python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-mocap --num_envs 16 \
#                                 --run_name "mocap" --max_iterations 700 \
#                                 --headless agent.algorithm.entropy_coef=0.1

# python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-mocap --num_envs 16 \
#                                 --run_name "mocap" --max_iterations 700 \
#                                 --headless agent.algorithm.entropy_coef=1.0

# python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-mocap --num_envs 16 \
#                                 --run_name "mocap" --max_iterations 700 \
#                                 --headless agent.algorithm.num_learning_epochs=10

# python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-mocap --num_envs 16 \
#                                 --run_name "mocap" --max_iterations 700 \
#                                 --headless agent.algorithm.num_learning_epochs=1

# python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-mocap --num_envs 16 \
#                                 --run_name "mocap" --max_iterations 700 \
#                                 --headless agent.algorithm.num_mini_batches=1

# python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-mocap --num_envs 16 \
#                                 --run_name "mocap" --max_iterations 700 \
#                                 --headless agent.algorithm.num_mini_batches=10

# python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-mocap --num_envs 16 \
#                                 --run_name "mocap" --max_iterations 700 \
#                                 --headless agent.algorithm.learning_rate=1.0e-3

# python scripts/rsl_rl/train.py --task Isaac-Vel-BALLU-mocap --num_envs 16 \
#                                 --run_name "mocap" --max_iterations 700 \
#                                 --headless agent.algorithm.max_grad_norm=1.0