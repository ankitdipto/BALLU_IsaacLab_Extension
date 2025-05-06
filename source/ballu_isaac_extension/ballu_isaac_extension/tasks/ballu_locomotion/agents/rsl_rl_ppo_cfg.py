from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass

@configclass
class BALLUPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 21 # Number of steps per environment before a policy update
    max_iterations = 1500 # Maximum number of policy upd iterations during training
    save_interval = 500
    experiment_name = "ballu_locomotion"
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
        max_grad_norm=0.5 # Maximum norm of the gradients for clipping
    )