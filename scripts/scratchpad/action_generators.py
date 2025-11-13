from isaaclab.envs.mdp.events import randomize_rigid_body_mass
from isaaclab.managers import SceneEntityCfg
import torch
import math

def get_periodic_action(step_count, period=200, num_envs=1):
    """Generate alternating motor actions for walking gait"""
    phase = (step_count % period) / period  # Normalized [0,1]
    
    # Sine wave pattern (amplitude 0.5π centered around π/2)
    #if step_count % 400 < 80:
    #    action_motor_left = 0.0
    #    action_motor_right = 0.0
    #else:
    dt_sim = 1 / 200.0
    control_time = step_count * dt_sim
    action_motor_left = 0.5 * math.sin(2 * math.pi * control_time) + 0.5
    action_motor_right = 0.5 * math.cos(2 * math.pi * control_time) + 0.5
    
    # Create tensor with shape (num_envs, 2)
    return torch.tensor(
        [[action_motor_left, action_motor_right] for _ in range(num_envs)],
        device="cuda:0"
    )

def stepper(step_count, period=200, num_envs=1):
    """
    Stepper controller for joint actuation.
    """
    actions = torch.full((num_envs, 2), 0.0, device="cuda:0")
    if step_count % period < period / 2:
        actions[:, 0] = 1.0
    else:
        actions[:, 1] = 1.0
    return actions

def bang_bang_control(step_count, num_envs=1):
    """
    Bang-bang controller for joint actuation.
    Args:
        step_count (int): Current step count
    Returns:
        torch.Tensor: Control actions (num_envs, num_joints)
    """
    min_action = 0.0
    max_action = 1.0
    #actions = torch.zeros((num_envs, 2), device="cuda:0")
    actions = torch.full((num_envs, 2), max_action, device="cuda:0") if (step_count % 2 == 0) else torch.full((num_envs, 2), min_action, device="cuda:0")
    return actions

def left_leg_0_right_leg_1(num_envs=1):
    """
    Left leg 0, right leg 1 controller for joint actuation.
    """
    actions = torch.full((num_envs, 2), 0.0, device="cuda:0")
    actions[:, 1] = 1.0
    return actions

def left_leg_1_right_leg_0(num_envs=1):
    """
    Left leg 1, right leg 0 controller for joint actuation.
    """
    actions = torch.full((num_envs, 2), 0.0, device="cuda:0")
    actions[:, 0] = 1.0
    return actions

def both_legs_0(num_envs=1):
    """
    Both legs 0 controller for joint actuation.
    """
    actions = torch.full((num_envs, 2), 0.0, device="cuda:0")
    return actions

def both_legs_1(num_envs=1):
    """
    Both legs 1 controller for joint actuation.
    """
    actions = torch.full((num_envs, 2), 1.0, device="cuda:0")
    return actions

def both_legs_theta(theta,num_envs=1):
    """
    Both legs theta controller for joint actuation.
    """
    actions = torch.full((num_envs, 2), theta, device="cuda:0")
    return actions

def override_link_masses_with_randomizer(env, link_name, mass_range, operation="abs"):
    """
    Override mass of a specific link using Isaac Lab's built-in randomizer.
    
    Args:
        env: The Isaac Lab environment
        link_name: Name of the link/body to modify
        mass_range: Tuple (min_mass, max_mass) or fixed value
        operation: "abs" to set absolute value, "scale" to multiply, "add" to add
    """
    # Create scene entity config for the specific robot
    asset_cfg = SceneEntityCfg("robot", body_names=[link_name])
    
    # Get all environment indices
    env_ids = torch.arange(env.num_envs, device=env.device)
    
    # Use Isaac Lab's mass randomization function
    if isinstance(mass_range, (int, float)):
        mass_range = (mass_range, mass_range)  # Convert single value to range
    
    randomize_rigid_body_mass(
        env=env,
        env_ids=env_ids,
        asset_cfg=asset_cfg,
        mass_distribution_params=mass_range,
        operation=operation,
        distribution="uniform",
        recompute_inertia=True
    )
    
    print(f"Set mass of {link_name} using randomizer with range {mass_range}")

def override_link_masses(env, link_masses_dict):
    """
    Override masses of specific links in the robot.
    
    Args:
        env: The Isaac Lab environment
        link_masses_dict: Dictionary mapping link names to new masses
                         e.g., {"BALLOON": 0.5, "TIBIA_LEFT": 0.02}
    """
    robot = env.unwrapped.scene["robot"]
    
    # Get current masses (shape: num_envs, num_bodies)
    current_masses = robot.root_physx_view.get_masses()
    
    # Get body names to find indices
    body_names = robot.body_names
    print(f"Available body names: {body_names}")
    
    # Create a copy of current masses to modify
    new_masses = current_masses.clone()
    
    # Override specific link masses
    for link_name, new_mass in link_masses_dict.items():
        if link_name in body_names:
            body_idx = body_names.index(link_name)
            # Set new mass for all environments
            new_masses[:, body_idx] = new_mass
            print(f"Set mass of {link_name} (body {body_idx}) to {new_mass} kg")
        else:
            print(f"Warning: Link '{link_name}' not found in body names")
    
    # Apply the new masses to simulation
    env_indices = torch.arange(robot.num_instances, dtype=torch.int, device=robot.device)
    robot.root_physx_view.set_masses(new_masses, env_indices)
    
    # Verify the changes
    updated_masses = robot.root_physx_view.get_masses()
    print("Mass override verification:")
    for i, body_name in enumerate(body_names):
        print(f"  {body_name}: {updated_masses[0, i].item():.6f} kg")
