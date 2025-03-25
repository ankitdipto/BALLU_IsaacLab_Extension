"""Launch Isaac Sim Simulator first."""

import argparse
import torch
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate bipedal robots.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformerCfg

from ballu_isaac_extension.ballu_assets.ballu_config import BALLU_CFG

@configclass
class BALLUSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    ballu: ArticulationCfg = BALLU_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    robot_transforms = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/PELVIS",
        target_frames=[FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/(?!.*Looks.*).*")],
        debug_vis=False,
    )

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""

    BALLOON_BUOYANCY_MASS = 0.25 #0.202 * 6.0
    BALLOON_DRAG_COEFFICIENT = 0.4
    BALLOON_BUOYANCY_FORCE = torch.tensor([[0.0, 0.0, 9.81 * BALLOON_BUOYANCY_MASS]],
                                          device=sim.device)
    PERTURB_ON_BALLOON = torch.tensor([[0.5, 0.0, 0.0]],
                                   device=sim.device)
    # Define origins
    origin = torch.tensor([
        [0.0, 0.0, 0.0]
    ]).to(device=sim.device)

    # JOINTS: HIP_LEFT - 0, HIP_RIGHT - 1, KNEE_LEFT - 3, KNEE_RIGHT - 4
    joint_targets = torch.tensor([
        [0.15, -0.15, 0.0, 0.0],
        [-0.15, 0.15, 0.0, 0.0]
    ], device = sim.device)
    goal_selector_idx = 0

    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["ballu"]
    print("robot", robot)
    last_env_idx = scene.num_envs - 1
    #print("Last env robot", scene["ballu"][last_env_idx])
    print("Joint names", robot.joint_names)
    print("Default joint positions -- ", robot.data.default_joint_pos)
    print("Body names", robot.body_names)

    print("robot joint limits", robot.data.soft_joint_pos_limits)
    print("robot joint velocity limits", robot.data.soft_joint_vel_limits)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    root_link_lin_vel_history = []
    root_com_lin_vel_history = []
    # Setting external Buoyancy force on the robot
    robot.set_external_force_and_torque(forces=BALLOON_BUOYANCY_FORCE, torques=torch.zeros(1,3, device=sim.device), is_global=True, body_ids = [3])
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 1000 == 0:
            goal_selector_idx = (goal_selector_idx + 1) % 2

        # if count == 500:
        #     robot.set_external_force_and_torque(forces=BALLOON_BUOYANCY_FORCE + FORCE_ON_PELVIS, 
        #                                         torques=torch.zeros(1,3, device=sim.device), 
        #                                         is_global=True, 
        #                                         body_ids = [3])
        #if count > 500:
            #robot.set_joint_position_target(joint_targets[goal_selector_idx].view(1, -1), joint_ids = [0, 1, 3, 4])
            #robot.write_data_to_sim()

        
            # reset counter
            #count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            # root_state = robot.data.default_root_state.clone()
            # root_state[:, :3] += scene.env_origins
            # robot.write_root_pose_to_sim(root_state[:, :7])
            # robot.write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            # joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            # joint_pos += torch.rand_like(joint_pos) * 0.1
            # robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # # clear internal buffers
            # scene.reset()
            # print("[INFO]: Resetting robot state...")
        # -- write data to sim
        # I am setting a very high value state for the robot to see how the simulation behaves
        if count == 0:
            large_tensor = torch.tensor([1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2])\
                                    .repeat(scene.num_envs, 1).to(device=sim.device)

            robot.write_joint_state_to_sim(position = torch.zeros((scene.num_envs, 7), device = sim.device), 
                                                                  velocity = large_tensor)
        root_link_lin_vel_history.append(robot.data.root_lin_vel_w[0])
        root_com_lin_vel_history.append(robot.data.root_com_lin_vel_w[0])
        robot_body_lin_vel_w = robot.data.body_lin_vel_w # (num_instances, num_bodies, 3)
        robot_balloon_lin_vel_w = robot_body_lin_vel_w[:, 3, :]
        drag = -torch.sign(robot_balloon_lin_vel_w) * BALLOON_DRAG_COEFFICIENT * (robot_balloon_lin_vel_w ** 2)
        EXT_PERTURBATION = (count > 500) * PERTURB_ON_BALLOON
        total_force = BALLOON_BUOYANCY_FORCE.repeat(scene.num_envs, 1) +\
                      EXT_PERTURBATION.repeat(scene.num_envs, 1) +\
                      drag
        robot.set_external_force_and_torque(forces=total_force.unsqueeze(1), 
                                            torques=torch.zeros(1,3, device=sim.device), 
                                            is_global=True, 
                                            body_ids = [3])
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)

        # Print the position of the robot
        #root_state = robot.data.root_state_w
        #print("Robot position:", root_state[:, :3])
        # Print information from the sensors
        #print(scene["robot_transforms"])
        #print("relative transforms:", scene["robot_transforms"].data.target_pos_source)
    return count, root_com_lin_vel_history, root_link_lin_vel_history

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[0, 6, 2.25], target=[0.0, 0.0, 1.0])

    # Design scene
    scene_cfg = BALLUSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Play the simulation
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")
    print("Gravity vector:", sim_cfg.gravity)
    # Run the simulator
    count, root_com_lin_vel_history, root_link_lin_vel_history = run_simulator(sim, scene)
    print("The simulation ran for {} steps.".format(count))

    # Convert tensors to cpu and extract components
    com_vel = torch.stack(root_com_lin_vel_history).cpu()
    link_vel = torch.stack(root_link_lin_vel_history).cpu()
    
    # Create time array
    time = torch.arange(len(com_vel)) * sim_cfg.dt

    # Plot COM velocities
    fig1, ax1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax1[0].plot(time, com_vel[:, 0], label='vx')
    ax1[1].plot(time, com_vel[:, 1], label='vy')
    ax1[2].plot(time, com_vel[:, 2], label='vz')
    
    ax1[0].legend()
    ax1[1].legend()
    ax1[2].legend()
    ax1[2].set_xlabel('Time (s)')
    fig1.suptitle('Center of Mass Velocities')
    fig1.savefig('com_velocities.png')

    # Plot Link velocities
    fig2, ax2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax2[0].plot(time, link_vel[:, 0], label='vx')
    ax2[1].plot(time, link_vel[:, 1], label='vy')
    ax2[2].plot(time, link_vel[:, 2], label='vz')
    
    ax2[0].legend()
    ax2[1].legend()
    ax2[2].legend()
    ax2[2].set_xlabel('Time (s)')
    fig2.suptitle('Link Velocities')
    fig2.savefig('link_velocities.png')

    plt.close('all')



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
