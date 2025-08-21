import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot tibia endpoints (world frame) for env 0 from .npy file")
    parser.add_argument(
        "--npy",
        type=str,
        default="tibia_endpoints_world.npy",
        help="Path to tibia_endpoints_world.npy (shape: (T, num_envs, 2, 3))",
    )
    args = parser.parse_args()

    npy_path = os.path.abspath(args.npy)
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"File not found: {npy_path}")

    data = np.load(npy_path)
    if data.ndim != 4 or data.shape[2] != 2 or data.shape[3] != 3:
        raise ValueError(
            f"Unexpected array shape {data.shape}. Expected (T, num_envs, 2, 3)."
        )

    # Select environment 0: shape (T, 2, 3)
    env0 = data[:397, 0, :, :]
    left_z = env0[:, 0, 2]   # (T, 2)
    right_z = env0[:, 1, 2]  # (T, 2)

    #plt.figure(figsize=(8, 8))
    plt.plot(left_z, label="Left foot", color="tab:blue")
    plt.plot(right_z, label="Right foot", color="tab:orange")
    plt.axhline(y = 0.015, label = "y = 0.015", color="tab:green", linestyle="--")
    
    plt.title("Foot touchpoints (world frame) — Env 0 (Z)")
    plt.xlabel("timesteps")
    plt.ylabel("Z [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_dir = os.path.dirname(npy_path)
    out_path = os.path.join(out_dir, "tibia_endpoints_env0_z.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()


