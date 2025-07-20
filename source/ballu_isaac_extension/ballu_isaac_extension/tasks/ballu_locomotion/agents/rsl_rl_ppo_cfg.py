from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlMirrorSymmetryCfg

@configclass
class BALLUPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 25 # Horizon: Number of steps per environment before a policy update
    max_iterations = 1500
    save_interval = 100
    experiment_name = "lab_7.22.2025"
    empirical_normalization = False  # Obs norm uses running mean and std. 
                                    # If true, compute stats empirically
                                    # from the current batch of observations
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5, # Initial std of exploration noise added to actions
        actor_hidden_dims=[128, 64, 32],
        critic_hidden_dims=[128, 64, 32],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, # Weight of value loss in the total loss
        use_clipped_value_loss=True, # Whether to clip value function updates similar to policy updates
        clip_param=0.2, # Limit policy updates within 20% of the previous policy
        entropy_coef=0.01, # Weight for the entropy bonus that encourages exploration
        num_learning_epochs=5, # Number of passes through the collected data for each policy update.
        num_mini_batches=4, # Number of mini-batches to split the data into for each policy update.
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95, # Lambda parameter for GAE
        desired_kl=0.01, # Target KL divergence between the old and new policy for adaptive learning rate adjustment
        max_grad_norm=0.5, # Maximum norm of the gradients for clipping
        
        # Mirror Symmetry Loss Configuration (Yu et al. approach)
        mirror_symmetry_cfg=RslRlMirrorSymmetryCfg(
            enabled=True,  # Enable mirror symmetry loss
            weight=0.1,    # Weight for mirror symmetry loss term
            
            # Define symmetric joint pairs for observations
            # Observation structure: velocity_commands(2) + joint_pos(7) + joint_vel(7)
            # Joint order: [NECK, HIP_LEFT, HIP_RIGHT, KNEE_LEFT, KNEE_RIGHT, MOTOR_LEFT, MOTOR_RIGHT]
            # Total obs dim: 2 + 7 + 7 = 16
            symmetric_joint_pairs=[
                # Joint positions (indices 2-8): velocity_commands(2) + joint_pos(7)
                (3, 4),   # HIP_LEFT <-> HIP_RIGHT (indices 2+1, 2+2)
                (5, 6),   # KNEE_LEFT <-> KNEE_RIGHT (indices 2+3, 2+4)  
                (7, 8),   # MOTOR_LEFT <-> MOTOR_RIGHT (indices 2+5, 2+6)
                # Joint velocities (indices 9-15): velocity_commands(2) + joint_pos(7) + joint_vel(7)
                (10, 11), # HIP_LEFT_vel <-> HIP_RIGHT_vel (indices 9+1, 9+2)
                (12, 13), # KNEE_LEFT_vel <-> KNEE_RIGHT_vel (indices 9+3, 9+4)
                (14, 15), # MOTOR_LEFT_vel <-> MOTOR_RIGHT_vel (indices 9+5, 9+6)
            ],
            
            # Define symmetric observation indices (sign flip for lateral components)
            # For velocity commands: lin_vel_y should be flipped for symmetry
            symmetric_obs_indices=[
                1,  # lin_vel_y command (index 1 in velocity_commands)
                # Note: Add other lateral velocity/angular velocity indices if your observations include them
            ],
            
            # Define symmetric action pairs for motor joints
            # Actions are [MOTOR_LEFT, MOTOR_RIGHT]
            symmetric_action_pairs=[
                (-1, -2),  # MOTOR_LEFT <-> MOTOR_RIGHT
            ],
            
            # Define symmetric action indices (sign flip for actions)
            # For BALLU, both motors should have the same action for symmetric gait
            # so we don't need to flip signs, just ensure they're symmetric
            symmetric_action_indices=[
                (0, 1),  # MOTOR_LEFT <-> MOTOR_RIGHT
            ],
        )
    )