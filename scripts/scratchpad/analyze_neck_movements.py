import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_neck_data(jsonl_file_path):
    """
    Load neck joint position data from JSONL file.
    
    Args:
        jsonl_file_path (str): Path to the JSONL file
        
    Returns:
        tuple: (x_labels, data_lists) where x_labels are Key[1] values and 
               data_lists are the corresponding Value lists
    """
    x_labels = []
    data_lists = []
    
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            # Parse each JSON object
            json_obj = json.loads(line.strip())
            
            # Extract the single key-value pair
            for key, value in json_obj.items():
                # Key should be a list of length 3, Value should be a list of length 400
                if isinstance(key, str):
                    # If key is stored as string, convert it back to list
                    key = json.loads(key)
                
                x_labels.append(key[1])  # Second element of the key list
                data_lists.append(value)  # The 400-length value list
    
    return x_labels, data_lists

def create_boxplots(x_labels, data_lists, save_path=None):
    """
    Create boxplots for neck joint position data.
    
    Args:
        x_labels (list): List of x-axis labels (Key[1] values)
        data_lists (list): List of data arrays for boxplots
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(12, 8))
    
    # Create boxplots
    box_plot = plt.boxplot(data_lists, labels=x_labels, patch_artist=True)
    
    # Customize the plot
    plt.xlabel('Point of application of Buoyancy', fontsize=12)
    plt.ylabel('Joint Position', fontsize=12)
    plt.title('Neck Joint Movement Analysis', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Color the boxes
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_lists)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Rotate x-axis labels if there are many
    if len(x_labels) > 10:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # plt.show()

def main():
    """Main function to analyze neck movements."""
    # Path to the JSONL file
    jsonl_file = "neck_joint_pos_history.jsonl"
    
    # Check if file exists
    if not Path(jsonl_file).exists():
        print(f"Error: File '{jsonl_file}' not found in current directory.")
        print("Please make sure the file is in the same directory as this script.")
        return
    
    try:
        # Load the data
        print(f"Loading data from {jsonl_file}...")
        x_labels, data_lists = load_neck_data(jsonl_file)
        
        print(f"Loaded {len(data_lists)} datasets")
        print(f"Each dataset contains {len(data_lists[0]) if data_lists else 0} values")
        print(f"X-axis labels (Key[1] values): {x_labels}")
        
        # Create boxplots
        print("Creating boxplots...")
        create_boxplots(x_labels, data_lists, save_path="neck_movement_analysis.png")
        
        # Print some statistics
        print("\nDataset Statistics:")
        for i, (label, data) in enumerate(zip(x_labels, data_lists)):
            data_array = np.array(data)
            print(f"Key[1]={label}: Mean={data_array.mean():.4f}, "
                  f"Std={data_array.std():.4f}, "
                  f"Min={data_array.min():.4f}, "
                  f"Max={data_array.max():.4f}")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        print("Please check the file format and try again.")

if __name__ == "__main__":
    main()
