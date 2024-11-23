import argparse
import os
import time
import sys
from datetime import datetime


def main(args):
    print("=== Mock Program Starting ===\n")

    # Confirm Python version
    print(f"Python Version: {sys.version}\n")

    # Confirm environment variables
    print("Environment Variables:")
    for key, value in os.environ.items():
        print(f"  {key}: {value}")
    print()

    # Confirm working directory
    cwd = os.getcwd()
    print(f"Current Working Directory: {cwd}\n")

    # Confirm presence of expected files
    expected_files = ['main.py', 'requirements.txt']
    print("Checking for expected files:")
    for file in expected_files:
        exists = os.path.isfile(file)
        print(f"  {file}: {'Found' if exists else 'Missing'}")
    print()

    # Test imports
    print("Testing package imports...")
    try:
        import gymnasium as gym
        import wandb
        import torch
        import numpy as np
        import tensorflow as tf
        import cv2
        import imageio
        import matplotlib.pyplot as plt

        print("All packages imported successfully.\n")
    except ImportError as e:
        print(f"Import Error: {e}\n")
        sys.exit(1)

    setup(args)

    # Simulate some operations with the packages
    print("Simulating operations with imported packages...\n")
    tensor = torch.tensor([1, 2, 3])
    print(f"Created a PyTorch tensor: {tensor}")

    np_array = np.array([4, 5, 6])
    print(f"Created a NumPy array: {np_array}")

    tf_constant = tf.constant([7, 8, 9])
    print(f"Created a TensorFlow constant: {tf_constant.numpy()}")

    # Create a simple OpenCV image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(img, 'Test', (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite('test-output/opencv_test.png', img)
    print("Created an OpenCV test image: opencv_test.png")

    # Create a simple matplotlib plot
    plt.plot([1, 2, 3], [4, 5, 6])
    plt.title('Matplotlib Test Plot')
    plt.savefig('matplotlib_test.png')
    print("Created a Matplotlib test plot: matplotlib_test.png\n")

    # Test statements
    test_statements = [
        "Test statement 1: Docker is working!",
        "Test statement 2: Printing to console.",
        "Test statement 3: Saving to a text file.",
        "Test statement 4: Mock program running.",
        "Test statement 5: Docker setup successful."
    ]

    # Print and save test statements
    output_file = "test-output/test_output.txt"
    print(f"Writing test statements to {output_file}...\n")
    with open(output_file, "w") as f:
        for statement in test_statements:
            print(statement)
            f.write(statement + "\n")
            time.sleep(0.5)  # Simulate some processing time

    print("\n=== Mock Program Completed Successfully ===")

def setup(args):

    if args.hyperparameters_path is None:
        hyperparameters = {
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "epsilon_start": args.epsilon_start,
            "epsilon_end": args.epsilon_end,
            "epsilon_decay": args.epsilon_decay,
            "replay_buffer_size": args.replay_buffer_size,
            "batch_size": args.batch_size,
            "target_update_freq": args.target_update_freq,
            "max_gradient_norm": args.max_gradient_norm,
            "hidden_size": args.hidden_size,
            "start_episode_length": args.start_episode_length,
            "performance_threshold": args.performance_threshold,
            "episode_length_increment": args.episode_length_increment,
            "max_steps_per_episode": args.max_steps_per_episode,
            "max_total_steps": args.max_total_steps,
            "eval_episodes": args.eval_episodes,
            "eval_steps": args.eval_steps,
            "use_gpu": args.use_gpu
        }

        # print hyperparameters
        print("Hyperparameters:")
        for key, value in hyperparameters.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    # Set up argument parsing for command-line arguments
    parser = argparse.ArgumentParser(description="Run the RL agent training.")
    parser.add_argument(
        "--hyperparameters_path",
        type=str,
        default=None,
        help="Path to the hyperparameters JSON file",
    )

    parser.add_argument(
        "--model_save_path",
        type=str,
        default="trained_models/",
        help="Path to the directory where the finished model will be stored",
    )

    parser.add_argument(
        "--model_load_path",
        type=str,
        default=None,
        help="Path to the model file where the training should continue",
    )

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    default_experiment_name = "experiment_" + current_time
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=default_experiment_name,
        help="Name of the model file",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/agent_performance/" + default_experiment_name,
        help="Name of the log file for the training run",
    )

    parser.add_argument(
        "--use_gpu",
        type=bool,
        default=False,
        help="Whether to use the GPU for training",
    )

    # for hyperparameter parsing
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards")
    parser.add_argument("--epsilon_start", type=float, default=1.0,
                        help="Starting value of epsilon for ε-greedy policy")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="Minimum value of epsilon for ε-greedy policy")
    parser.add_argument("--epsilon_decay", type=float, default=0.99978,
                        help="Decay rate of epsilon for ε-greedy policy")
    parser.add_argument("--replay_buffer_size", type=int, default=10000, help="Size of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--target_update_freq", type=int, default=1000, help="Frequency of target network updates")
    parser.add_argument("--max_gradient_norm", type=float, default=1.0, help="Maximum norm for gradient clipping")
    parser.add_argument("--hidden_size", type=int, default=64, help="Number of hidden units in the Q-network")

    parser.add_argument("--start_episode_length", type=int, default=100, help="Initial length of each episode")
    parser.add_argument("--performance_threshold", type=float, default=0.5,
                        help="Performance threshold for adjusting episode length")
    parser.add_argument("--episode_length_increment", type=int, default=100,
                        help="Increment for episode length when performance threshold is met")
    parser.add_argument("--max_steps_per_episode", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--max_total_steps", type=int, default=1000, help="Maximum total steps for the experiment")
    parser.add_argument("--eval_episodes", type=int, default=2, help="Number of episodes to evaluate the model")
    parser.add_argument("--eval_steps", type=int, default=400, help="Number of steps per evaluation episode")
    arg_parser = parser.parse_args()

    main(arg_parser)
