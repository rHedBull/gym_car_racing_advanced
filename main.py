import json
import gymnasium as gym
from datetime import datetime
from Agent import Agent

from ExperimentLogger import ExperimentLogger
from train import train, evaluate_agent

eval_episodes = 10

def main():
    env = gym.make("CarRacing-v2", render_mode="rgb-array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/agent_performance/" + current_time + "_stats"
    model_path = "models/" + current_time + "_model"
    hyperparameters_path = "hyperparameters.json"
    logger = ExperimentLogger(log_dir)
    run(env, logger, model_path, hyperparameters_path)



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
    main()
    #eval_model()
