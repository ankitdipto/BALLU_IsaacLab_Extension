import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def plot_joint_data(joint_pos_hist_tch, joint_vel_hist_tch, joint_names, num_envs, save_dir):
    """Plot joint positions and velocities for each environment.
    
    Args:
        joint_pos_hist_tch: Tensor of joint positions history [timesteps, envs, joints]
        joint_vel_hist_tch: Tensor of joint velocities history [timesteps, envs, joints]
        joint_names: List of joint names
        num_envs: Number of environments
        save_dir: Directory to save plots
    """
    for env_idx in range(num_envs):
        plt.figure(figsize=(20, 6))
        
        # Define a list of distinguishable colors
        # Using a common palette for good visual separation
        colors = [
            '#1f77b4',  # Muted blue
            '#ff7f0e',  # Safety orange
            '#2ca02c',  # Cooked asparagus green
            '#d62728',  # Brick red
            '#9467bd',  # Muted purple
            '#8c564b',  # Chestnut brown
            '#e377c2',  # Raspberry yogurt pink
            # Add more if more than 7 joints are ever needed, 
            # though the request specifies 7
            '#7f7f7f',  # Middle gray
            '#bcbd22',  # Curry yellow-green
            '#17becf'   # Blue-teal
        ]

        # Position plot
        for joint_idx in range(joint_pos_hist_tch.shape[2]):
            plt.plot(joint_pos_hist_tch[:, env_idx, joint_idx], label=f'{joint_names[joint_idx]}', color=colors[joint_idx % len(colors)], linewidth=2.0)
        plt.title(f'Joint Positions Over Time (Env {env_idx})')
        plt.xlabel('Timestep')
        plt.ylabel('Position (rad)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"joint_plots_env_{env_idx}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"[INFO] Saved joint plots for environment {env_idx} to {plot_path}")

def plot_root_com_xy(root_com_xyz_hist_tch, num_envs, save_dir):
    """Plot root COM X,Y positions for all environments in a single figure.
    
    Args:
        root_com_xyz_hist_tch: Tensor of root COM XYZ history [timesteps, envs, 3]
        num_envs: Number of environments
        save_dir: Directory to save plots
    """
    # Calculate figure size based on number of environments
    grid_size = int(np.ceil(np.sqrt(num_envs)))
    fig_size = max(8, grid_size * 4)  # Minimum 8 inches, scale with env count
    
    plt.figure(figsize=(fig_size, fig_size))
        
    # Extract all X,Y coordinates
    all_xy = np.array([[pos[env_idx][:2] for pos in root_com_xyz_hist_tch] 
                      for env_idx in range(num_envs)])
        
    # Plot each environment's trajectory (swap axes, invert x for Y-left, X-up)
    for env_idx in range(num_envs):
        plt.plot(all_xy[env_idx][:, 1], all_xy[env_idx][:, 0], 
                label=f'Env {env_idx}', alpha=0.7)

    # Enforce equal scaling on X and Y axes
    ax = plt.gca()
    x_vals = all_xy[:, :, 1].reshape(-1)
    y_vals = all_xy[:, :, 0].reshape(-1)
    x_min, x_max = np.nanmin(x_vals), np.nanmax(x_vals)
    y_min, y_max = np.nanmin(y_vals), np.nanmax(y_vals)
    x_c, y_c = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    half_span = max(x_max - x_min, y_max - y_min) / 2.0
    # In case of degenerate span, use a tiny epsilon to avoid zero range
    if half_span == 0:
        half_span = 1e-6
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_c - half_span, x_c + half_span)
    ax.set_ylim(y_c - half_span, y_c + half_span)
    ax.invert_xaxis()  # Invert X so positive Y points left
    
    plt.title(f'Root COM (Y,X) Positions (X↑, Y←) (All {num_envs} Environments)')
    plt.xlabel('Y Position')  # Now horizontal, but inverted
    plt.ylabel('X Position')  # Now vertical
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
        
    plot_path = os.path.join(save_dir, "root_com_xy_positions_all.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[INFO] Saved combined root COM XY plot to {plot_path}")

def plot_feet_heights(left_foot_pos_history, right_foot_pos_history, num_envs, save_dir):
    """Plot feet heights (Z) over time for all environments as subplots in a single figure.
    
    Args:
        left_foot_pos_history: List of left foot position tensors
        right_foot_pos_history: List of right foot position tensors
        num_envs: Number of environments
        save_dir: Directory to save plots
    """
    # Skip if foot position history is empty or contains None values
    if not left_foot_pos_history or not right_foot_pos_history or None in left_foot_pos_history or None in right_foot_pos_history:
        print("[INFO] Skipping feet height plots due to missing data")
        return
        
    left_foot_np = torch.cat(left_foot_pos_history, dim=0).cpu().numpy()  # shape: (timesteps, envs, 3)
    right_foot_np = torch.cat(right_foot_pos_history, dim=0).cpu().numpy()
    timesteps = left_foot_np.shape[0]

    # Dynamic figure size: width=5 per env, height=3 per env (up to a reasonable max)
    fig_width = max(8, min(5 * num_envs, 30))
    fig_height = max(3, min(2 * num_envs, 20))
    fig, axs = plt.subplots(num_envs, 1, figsize=(fig_width, fig_height), squeeze=False)
    
    for env_idx in range(num_envs):
        ax = axs[env_idx, 0]
        ax.plot(range(timesteps), left_foot_np[:, env_idx, 2], label='Left Foot', color='blue', alpha=0.7)
        ax.plot(range(timesteps), right_foot_np[:, env_idx, 2], label='Right Foot', color='red', alpha=0.7)
        ax.set_ylabel('Z (m)')
        ax.set_title(f'Env {env_idx}')
        ax.grid(True)
        ax.legend()
    
    axs[-1, 0].set_xlabel('Timestep')
    plt.suptitle('Feet Heights (Z) Over Time (All Environments)')
    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    
    feet_plot_path = os.path.join(save_dir, 'feet_heights_z_all_envs.png')
    plt.savefig(feet_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[INFO] Saved feet heights Z plot to {feet_plot_path}")

def plot_base_velocity(base_vel_hist_tch, num_envs, save_dir):
    """Plot base velocity over time for all environments as subplots in a single figure.
    
    Args:
        base_vel_hist_tch: Tensor of base velocity history [timesteps, envs, 3]
        num_envs: Number of environments
        save_dir: Directory to save plots
    """
    # Convert to numpy for plotting
    base_vel_np = base_vel_hist_tch.cpu().numpy()  # shape: (timesteps, envs, 3)
    timesteps = base_vel_np.shape[0]
    
    # Component labels
    component_labels = ['X (Forward)', 'Y (Lateral)', 'Z (Vertical)']
    component_colors = ['#4878CF', '#6ACC65', '#D65F5F']  # Muted blue, green, red
    
    # Calculate grid dimensions for a balanced layout
    n_cols = int(np.ceil(np.sqrt(num_envs)))
    n_rows = int(np.ceil(num_envs / n_cols))
    
    # Dynamic figure size: based on grid dimensions
    fig_width = max(4 * n_cols, 8)  # At least 4 inches per column, minimum 8 inches
    fig_height = max(3 * n_rows, 6)  # At least 3 inches per row, minimum 6 inches
    
    # Create figure with subplots in a grid layout
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    
    # Plot each environment in a separate subplot
    env_idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if env_idx < num_envs:
                ax = axs[row, col]
                
                # Plot each velocity component
                for comp_idx in range(3):
                    ax.plot(
                        range(timesteps), 
                        base_vel_np[:, env_idx, comp_idx], 
                        label=component_labels[comp_idx], 
                        color=component_colors[comp_idx], 
                        alpha=0.7,  # Slightly reduce opacity
                        linewidth=1.5  # Slightly thicker lines for better visibility with muted colors
                    )
                
                ax.set_ylabel('Velocity (m/s)')
                ax.set_xlabel('Timestep')
                ax.set_title(f'Env {env_idx}')
                ax.grid(True)
                
                # Only add legend to the first subplot to save space
                if row == 0 and col == 0:
                    ax.legend(loc='upper right')
                
                env_idx += 1
            else:
                # Hide unused subplots
                axs[row, col].set_visible(False)
    
    # Add a single legend for the entire figure
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=3)
    
    # Remove the individual legend from the first subplot since we have a common one
    if axs[0, 0].get_legend():
        axs[0, 0].get_legend().remove()
    
    # Add overall title with more space
    plt.suptitle('Base Velocity Over Time (All Environments)', y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.92))  # Increase top margin for title and legend
    
    # Save the figure
    vel_plot_path = os.path.join(save_dir, 'base_velocity_all_envs.png')
    plt.savefig(vel_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[INFO] Saved base velocity plot to {vel_plot_path}")

def plot_knee_phase_portraits(joint_pos_hist_tch, joint_vel_hist_tch, joint_names, num_envs, save_dir):
    """Create phase portraits (position vs. velocity) for knee joints across all environments.
    
    Args:
        joint_pos_hist_tch: Tensor of joint positions history [timesteps, envs, num_joints]
        joint_vel_hist_tch: Tensor of joint velocities history [timesteps, envs, num_joints]
        joint_names: List of joint names
        num_envs: Number of environments
        save_dir: Directory to save plots
    """
    # Convert to numpy for plotting
    joint_pos_np = joint_pos_hist_tch.cpu().numpy()  # shape: (timesteps, envs, num_joints)
    joint_vel_np = joint_vel_hist_tch.cpu().numpy()  # shape: (timesteps, envs, num_joints)
    
    # Find indices for knee joints
    knee_indices = []
    knee_names = []
    for i, name in enumerate(joint_names):
        if "KNEE_LEFT" in name or "KNEE_RIGHT" in name:
            knee_indices.append(i)
            knee_names.append(name)
    
    if not knee_indices:
        print("[INFO] No knee joints found, skipping phase portrait")
        return
    
    # We need one row per environment and two columns (left and right knee)
    n_rows = num_envs
    n_cols = len(knee_indices)  # Should be 2 for left and right knees
    
    # Create figure
    fig_width = max(12, 6 * n_cols)
    fig_height = max(10, 3 * n_rows)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    
    # Colors for different environments - use a different color for each environment
    env_colors = plt.cm.viridis(np.linspace(0, 1, num_envs))
    
    # Plot phase portraits for each environment and knee joint
    for env_idx in range(num_envs):
        for col, (joint_idx, joint_name) in enumerate(zip(knee_indices, knee_names)):
            # Get position and velocity data for this joint and environment
            pos = joint_pos_np[:, env_idx, joint_idx]
            vel = joint_vel_np[:, env_idx, joint_idx]
            
            # Plot phase portrait
            axs[env_idx, col].plot(pos, vel, color=env_colors[env_idx], alpha=0.8)
            axs[env_idx, col].set_xlabel('Joint Position (rad)')
            axs[env_idx, col].set_ylabel('Joint Velocity (rad/s)')
            axs[env_idx, col].set_title(f'{joint_name} - Env {env_idx}')
            axs[env_idx, col].grid(True)
            
            # Add arrows to show direction of trajectory
            n_arrows = 10
            step = max(1, len(pos) // n_arrows)
            for i in range(0, len(pos) - 1, step):
                if i + 1 < len(pos):  # Ensure we don't go out of bounds
                    axs[env_idx, col].annotate('', 
                        xytext=(pos[i], vel[i]),
                        xy=(pos[i+1], vel[i+1]), 
                        arrowprops=dict(arrowstyle='->', color=env_colors[env_idx], lw=1.5),
                        annotation_clip=True)
    
    plt.suptitle('Knee Joint Phase Portraits (Position vs. Velocity)', y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.96))  # Make room for the title
    
    # Save the figure
    phase_plot_path = os.path.join(save_dir, 'knee_phase_portraits.png')
    plt.savefig(phase_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[INFO] Saved knee joint phase portraits to {phase_plot_path}")

def plot_toe_heights(tibia_endpoints_hist_tch, num_envs, save_dir):
    """Plot toe heights (Z) over time for all environments as subplots in a single figure.
    
    Args:
        tibia_endpoints_hist_tch: Tensor of toe endpoints world positions [timesteps, envs, 2, 3]
        num_envs: Number of environments
        save_dir: Directory to save plots
    """
    if tibia_endpoints_hist_tch is None or tibia_endpoints_hist_tch.numel() == 0:
        print("[INFO] Skipping toe height plots due to missing data")
        return
    # Convert to numpy for plotting
    toe_np = tibia_endpoints_hist_tch.cpu().numpy()  # shape: (T, E, 2, 1, 3)
    toe_np = toe_np.squeeze(3)
    print("toe_np.shape: ", toe_np.shape)
    timesteps = toe_np.shape[0]
    
    # Calculate grid dimensions for a balanced layout
    n_cols = int(np.ceil(np.sqrt(num_envs)))
    n_rows = int(np.ceil(num_envs / n_cols))
    
    # Dynamic figure size
    fig_width = max(4 * n_cols, 8)
    fig_height = max(3 * n_rows, 6)
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    
    env_idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if env_idx < num_envs:
                ax = axs[row, col]
                # Left toe (index 0), Right toe (index 1)
                ax.plot(range(timesteps), toe_np[:, env_idx, 0, 2], label='Left Toe Z', color='#1f77b4', alpha=0.8, linewidth=1.5)
                ax.plot(range(timesteps), toe_np[:, env_idx, 1, 2], label='Right Toe Z', color='#d62728', alpha=0.8, linewidth=1.5)
                ax.set_ylabel('Height Z (m)')
                ax.set_xlabel('Timestep')
                ax.set_title(f'Env {env_idx}')
                ax.grid(True)
                # Only add legend to the first subplot
                if row == 0 and col == 0:
                    ax.legend(loc='upper right')
                env_idx += 1
            else:
                axs[row, col].set_visible(False)
    
    # Add a single legend for the entire figure
    handles, labels = axs[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=2)
    
    plt.suptitle('Toe Heights (Z) Over Time (All Environments)', y=0.99)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    
    plot_path = os.path.join(save_dir, 'toe_heights_all_envs.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[INFO] Saved toe heights plot to {plot_path}")


def plot_local_positions_scatter(local_positions_tch, save_dir, threshold_x=1.0, success_rate: float | None = None):
    """Scatter plot of final local positions (X vs Y) for all environments.
    
    Args:
        local_positions_tch: Tensor of shape (num_envs, 3) with local XYZ positions.
        save_dir: Directory to save the plot.
        threshold_x: Success threshold along local X (m). Default: 1.7.
    """
    if local_positions_tch is None or local_positions_tch.numel() == 0:
        print("[INFO] Skipping local positions scatter due to missing data")
        return

    lp = local_positions_tch.detach().cpu().numpy()  # (E, 3)
    x = lp[:, 0]
    y = lp[:, 1]

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, c="#1f77b4", alpha=0.85, edgecolors="k", linewidths=0.5, label="Env final pos")
    plt.axvline(threshold_x, color="#d62728", linestyle="--", linewidth=2.0, label=f"x = {threshold_x} m")

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    # ax.invert_xaxis()  # Flip x-axis so +x is on the left side
    plt.xlabel('Local X (m)')
    plt.ylabel('Local Y (m)')
    title = 'Final Local Positions (XY)'
    if success_rate is not None:
        title += f'  |  Success Rate: {success_rate:.2%}'
    plt.title(title)
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()

    plot_path = os.path.join(save_dir, 'final_local_positions_scatter.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[INFO] Saved local positions scatter to {plot_path}")

