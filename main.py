import argparse
import json
from datetime import datetime

import wandb
import gymnasium as gym

from Agent import Agent
from ExperimentLogger import ExperimentLogger
from train import evaluate_agent, train
from DQN import load_model




def main(args):

    hyperparameters, old_model_data, logger = setup(args)



    env = gym.make(
        "CarRacing-v2",
        render_mode="rgb-array",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
    )


    full_path = args.model_save_path + logger.experiment_name + ".pth"
    agent = Agent(hyperparameters, logger, old_model_data)
    wandb.config.update(hyperparameters, allow_val_change=True)

    train(env, agent, logger, hyperparameters)
    agent.save_model(full_path, log_to_wandb=True, artifact_name="finished_model")
    evaluate_agent(agent, logger, hyperparameters.get("eval_episodes"), hyperparameters.get("eval_steps"), True)

    logger.close()
    env.close()

    wandb.finish()


def setup(args):

    # for testing
    args.hyperparameters_path = "hyperparameters.json"

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
        }
    else:
        with open(args.hyperparameters_path, "r") as f:
            hyperparameters = json.load(f)

    logger = ExperimentLogger(args.log_dir, args.experiment_name)

    old_model_data = None
    if args.model_load_path is not None:
        old_model_data = load_model(args.model_load_path)
        logger.experiment_name = old_model_data["experiment_name"]


        wandb.init(
            project="gymnasium_car_racing",  # Replace with your project name
            name=logger.experiment_name,
            config=hyperparameters,
            mode="online",
            resume="allow",
            id=old_model_data["wandb_run_id"],
        )
    else:
        wandb.init(
            project="gymnasium_car_racing",  # Replace with your project name
            name=logger.experiment_name,
            config=hyperparameters,
            mode="online"
        )

    config = wandb.config

    return hyperparameters, old_model_data, logger


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
    arg_parser = parser.parse_args()

    main(arg_parser)
    # eval_model()
