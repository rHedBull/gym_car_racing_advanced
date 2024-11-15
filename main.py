import argparse
import json
import gymnasium as gym
from datetime import datetime
from Agent import Agent

from ExperimentLogger import ExperimentLogger
from train import train, evaluate_agent

eval_episodes = 10

def main(args):
    env = gym.make("CarRacing-v2", render_mode="rgb-array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)

    logger = ExperimentLogger(args.log_dir, args.experiment_name)
    full_path = args.model_dir_path + args.experiment_name + ".pth"
    run(env, logger, full_path, args.hyperparameters_path)



def run(env, logger, model_path, hyperparameters):

    with open(hyperparameters, "r") as f:
        hyperparameters = json.load(f)
    agent = Agent(hyperparameters, logger)

    train(env, agent, logger)
    agent.save_model(model_path)
    evaluate_agent(agent, env, logger, 10, eval_episodes)

    logger.close()
    env.close()

if __name__ == "__main__":
    # Set up argument parsing for command-line arguments
    parser = argparse.ArgumentParser(description="Run the RL agent training.")
    parser.add_argument(
        "--hyperparameters_path",
        type=str,
        default="./hyperparameters.json",
        help="Path to the hyperparameters JSON file",
    )

    parser.add_argument(
        "--model_dir_path",
        type=str,
        default="models/",
        help="Path to the directory where the finished model will be stored",
    )
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    default_experiment_name = "experiment_" + current_time + ".pth"
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
    arg_parser = parser.parse_args()

    main(arg_parser)
    #eval_model()
