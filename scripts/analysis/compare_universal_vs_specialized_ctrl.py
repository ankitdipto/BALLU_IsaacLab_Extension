import os
import sys
import random
import re
import numpy as np
from datetime import datetime
import subprocess
import json
import argparse
import cma
import torch
import csv
from typing import Dict, List, Tuple, Optional

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
ext_dir = os.path.join(project_dir, "source", "ballu_isaac_extension", "ballu_isaac_extension")
if ext_dir not in sys.path:
    sys.path.insert(0, ext_dir)

def run_evaluation_experiment(
        morph_id: str,
        task: str = "Isc-BALLU-hetero-general",
        load_run: str = "",
        checkpoint: str = "model_best.pt",
        num_envs: int = 64,
        num_episodes: int = 30,
        spring_coeff: float = 0.00807,
        gravity_comp_ratio: float = 0.65,
        difficulty_level: int = -1
    ) -> tuple[bool, float, str]:
    """Run evaluation with universal controller and return success status, curriculum level, and eval info."""
    eval_script_path = f"{project_dir}/scripts/rsl_rl/evaluate_design_cmaes.py"
    cmd = [
        sys.executable,
        eval_script_path,
        "--task", task,
        "--load_run", load_run,
        "--checkpoint", checkpoint,
        "--num_envs", str(num_envs),
        "--num_episodes", str(num_episodes),
        "--GCR", str(gravity_comp_ratio),
        "--spcf", str(spring_coeff),
        "--difficulty_level", str(difficulty_level),
        "--headless",
        "agent.experiment_name=lab_11.18.2025"
    ]

    env = os.environ.copy()
    env['BALLU_USD_REL_PATH'] = morph_id
    env['ISAAC_SIM_PYTHON_EXE'] = sys.executable
    env['FORCE_GPU'] = '1'

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

    curriculum_level = None
    FL = None
    TL = None
    eval_info = ""
    try:
        for line in process.stdout:
            print(line, end='')
            eval_info += line
            # Parse BEST_CRCLM_LEVEL from output
            if line.startswith("BEST_CRCLM_LEVEL:"):
                try:
                    curriculum_level = float(line.split(":")[1].strip())
                except (ValueError, IndexError) as e:
                    print(f"[WARNING] Failed to parse curriculum level: {e}")
            
            if line.startswith("FL:"):
                try:
                    FL = float(line.split(":")[1].strip())
                except (ValueError, IndexError) as e:
                    print(f"[WARNING] Failed to parse FL: {e}")
            
            if line.startswith("TL:"):
                try:
                    TL = float(line.split(":")[1].strip())
                except (ValueError, IndexError) as e:
                    print(f"[WARNING] Failed to parse TL: {e}")

        process.wait(timeout=30 * 60)  # 30 minute timeout
    except subprocess.TimeoutExpired:
        print(f"Evaluation timeout (30 min), killing process")
        process.kill()
        process.wait()
        return False, float('-inf'), "", -1.0, -1.0

    if process.returncode != 0:
        return False, float('-inf'), "", -1.0, -1.0
    
    if curriculum_level is None:
        print(f"[WARNING] Could not extract curriculum level from output")
        curriculum_level = float('-inf')
    
    # Use -1.0 as default if FL or TL were not parsed
    if FL is None:
        FL = -1.0
    if TL is None:
        TL = -1.0
    
    return True, curriculum_level, eval_info, FL, TL

def run_testing_experiment(
        morph_id: str, 
        task: str = "Isc-Vel-BALLU-1-obstacle",
        load_run: str = "",
        checkpoint: str = "model_best.pt",
        num_envs: int = 1,
        video_length: int = 399,
        device: str = "cuda:0",
        difficulty_level: int = 0,
        spring_coeff: float = 0.00807,
        gravity_comp_ratio: float = 0.65,
        cmdir: str = "test"
    ) -> tuple[bool, str]:
    """Run testing experiment for a trained morphology and return success status."""
    test_script_path = f"{project_dir}/scripts/rsl_rl/play_universal.py"
    cmd = [
        sys.executable,
        test_script_path,
        "--task", task,
        "--load_run", load_run,
        "--checkpoint", checkpoint,
        "--num_envs", str(num_envs),
        "--video",
        "--video_length", str(video_length),
        "--device", device,
        "--headless",
        "--difficulty_level", str(difficulty_level),
        "--GCR", str(gravity_comp_ratio),
        "--spcf", str(spring_coeff),
        "--cmdir", cmdir,
    ]

    env = os.environ.copy()
    env['BALLU_USD_REL_PATH'] = morph_id
    env['ISAAC_SIM_PYTHON_EXE'] = sys.executable
    env['FORCE_GPU'] = '1'

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )

    try:
        for line in process.stdout:
            print(line, end='')
        process.wait(timeout=2 * 3600)
    except subprocess.TimeoutExpired:
        print(f"Testing timeout (2h), killing process")
        process.kill()
        process.wait()
        return False, ""

    test_command = f"env BALLU_USD_REL_PATH={env['BALLU_USD_REL_PATH']}" + " " + ' '.join(cmd)
    return process.returncode == 0, test_command

def run_training_experiment(
        morph_id: str, 
        task: str = "Isc-Vel-BALLU-1-obstacle",
        max_iterations: int = 50,
        seed: int = 42,
        spring_coeff: float = 0.00807,
        gravity_comp_ratio: float = 0.65
    ) -> tuple[bool, float, str]:
    """Run training experiment and return success status and best curriculum level."""
    train_script_path = f"{project_dir}/scripts/rsl_rl/train.py"
    # extract morphology name from morph_id
    morphology_name = morph_id.split("/")[-1].split(".")[0]
    cmd = [
        sys.executable, 
        train_script_path,
        "--task", task,
        "--num_envs", "4096",
        "--max_iterations", str(max_iterations),
        "--run_name", morphology_name,
        "--headless",
        "--seed", str(seed),
        "--GCR", str(gravity_comp_ratio),
        "--spcf", str(spring_coeff)
    ]

    env = os.environ.copy()
    env['BALLU_USD_REL_PATH'] = morph_id
    env['ISAAC_SIM_PYTHON_EXE'] = sys.executable
    env['FORCE_GPU'] = '1'

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

    log_dir = None
    try:
        for line in process.stdout:
            print(line, end='')
            if line.startswith("EXP_DIR:"):
                log_dir = line.split("EXP_DIR:")[1].strip()
        process.wait(timeout=2 * 3600)
    except subprocess.TimeoutExpired:
        print(f"Training timeout (2h), killing process")
        process.kill()
        process.wait()
        return False, float('-inf'), ""

    if process.returncode != 0:
        return False, float('-inf'), ""
    
    ckpt_dict = torch.load(os.path.join(log_dir, "model_best.pt"))
    best_crclm_level = ckpt_dict["best_crclm_level"]
    
    return True, best_crclm_level, log_dir

def main():

    TASK = "Isc-BALLU-hetero-general"
    SPCF = 0.01
    GCR = 0.85
    # EVAL_DIFFICULTY_LEVEL = 20
    # NUM_EPISODES = 20
    # LOAD_RUN = "2025-11-18_09-33-39_universal_dynamic_r2s_v1"
    
    morphology_library = "source/ballu_isaac_extension/ballu_isaac_extension/ballu_assets/robots/morphologies/hetero_library_20251118_090817"
    morphologies = os.listdir(morphology_library)

    # print(f"Morphologies: {morphologies}")

    # Create CSV file with headers
    # csv_filename = os.path.join(script_dir, "evaluation_results_gcr_spcf_hetero_0006.csv")
    # with open(csv_filename, 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['GCR', 'SPCF', 'obstacle_height'])

    # for morphology_name in morphologies:
    for morph_name in ['hetero_0000', 'hetero_0001', 'hetero_0002', 'hetero_0003', 'hetero_0004', 
    'hetero_0005', 'hetero_0006', 'hetero_0007', 'hetero_0008', 'hetero_0009', 'hetero_0010', 'hetero_0011', 
    'hetero_0012', 'hetero_0013', 'hetero_0014', 'hetero_0015', 'hetero_0016', 'hetero_0017', 'hetero_0018', 'hetero_0019', 'hetero_0020']:
        morph_id = os.path.join("morphologies/hetero_library_20251118_090817", morph_name, morph_name + ".usd")
        print(f"Morphology: {morph_id}, GCR: {GCR}, SPCF: {SPCF}")
        success, best_crclm_level, log_dir = run_training_experiment(
            morph_id,
            task=TASK,
            spring_coeff=SPCF,
            gravity_comp_ratio=GCR,
            max_iterations=1600,
        )
        run_name = log_dir.split("/")[-1] if log_dir else ""
        difficulty = int(best_crclm_level * 100) - 1 if best_crclm_level > 0 else 0
        test_success, test_command = run_testing_experiment(
            morph_id,
            task=TASK,
            load_run=run_name,
            checkpoint="model_best.pt",
            difficulty_level=difficulty,
            spring_coeff=SPCF,
            gravity_comp_ratio=GCR,
            cmdir="tests"
        )
        print("#" * 60)
        print(f"Training status: {success}")
        print(f"Best curriculum level: {best_crclm_level}")
        print(f"Test status: {test_success}")
        print("#" * 60)
        # success, curriculum_level, eval_info, FL, TL = run_evaluation_experiment(
        #     morph_id,
        #     task=TASK,
        #     spring_coeff=spcf,
        #     gravity_comp_ratio=gcr,
        #     difficulty_level=EVAL_DIFFICULTY_LEVEL,
        #     num_episodes=NUM_EPISODES,
        #     load_run=LOAD_RUN
        # )
        # print(f"Success: {success}")
        # print(f"Curriculum Level: {curriculum_level}")
        # print(f"Eval Info: {eval_info}")
        # # print(f"FL: {FL}")
        # # print(f"TL: {TL}")
        
        # # Write results to CSV file
        # with open(csv_filename, 'a', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow([gcr, spcf, curriculum_level])
        
        # print(f"Results written to {csv_filename}")

if __name__ == "__main__":
    main()