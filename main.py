import gymnasium as gym
from datetime import datetime

from Agent import Agent
from ExperimentLogger import ExperimentLogger

num_episodes = 10

start_episode_length = 100
performance_threshold = 10
episode_length_increment = 100
max_steps_per_episode = 100

eval_episodes = 30

cut_off_reward = -1000

action_order = ["steering", "gas", "breaking"]

def main():
    env = gym.make("CarRacing-v2", render_mode="rgb-array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/agent_performance/" + current_time + "_stats"
    model_path = "models/" + current_time + "_model"
    logger = ExperimentLogger(log_dir)
    run(env, logger, model_path)

def train(env, agent, logger):
    steps_per_episode = start_episode_length
    total_steps_taken = 0

    for episode in range(num_episodes):

        # run the game loop
        step = 0
        done = False
        info = env.reset()
        old_observation = info[0]
        new_observation = None

        # for metrics logging
        episode_rewards = []

        while not done and step < steps_per_episode:
            if env.render_mode == "human":
                env.render()

            # if total_reward < cut_off_reward:
            #     continue

            action = agent.get_action(old_observation)

            new_observation, reward, _, all_done, _ = env.step(action)
            agent.store_transition(old_observation, action, reward, new_observation, all_done)

            agent.train()

            episode_rewards.append(reward)
            old_observation = new_observation
            step += 1

        total_steps_taken += step
        total_episode_reward = sum(episode_rewards)
        average_reward = total_episode_reward / len(episode_rewards)

        logger.log_episode_metrics(episode, step, total_episode_reward)

        print(
            f"Episode {episode + 1}: Total Reward = {total_episode_reward}, Avrg. Reward = {average_reward}, Steps = {step}, Epsilon = {agent.DQN.epsilon:.4f}")

        steps_per_episode = adjust_max_steps(average_reward, performance_threshold, step, episode_length_increment)

    print("TRAINING COMPLETE")
    print(f"Total steps taken: {total_steps_taken}")

def run(env, logger, model_path):

    agent = Agent(logger)

    train(env, agent, logger)
    agent.save_model(model_path)
    evaluate_agent(agent, env, logger)

    logger.close()
    env.close()


def evaluate_agent(agent, env, logger, num_episodes=10, eval_steps=200):

    original_epsilon = agent.DQN.epsilon
    agent.DQN.epsilon = 0.0  # Disable exploration

    print("STARTING EVALUATION")
    for episode in range(num_episodes):

        # run the game loop
        step = 0
        done = False
        info = env.reset()
        old_observation = info[0]
        new_observation = None

        total_steps_taken = 0

        # for metrics logging
        episode_rewards = []

        while not done and step < eval_steps:
            if env.render_mode == "human":
                env.render()

            action = agent.get_action(old_observation)

            new_observation, reward, _, all_done, _ = env.step(action)

            episode_rewards.append(reward)

            old_observation = new_observation
            step += 1

        total_steps_taken += step
        total_episode_reward = sum(episode_rewards)
        average_reward = total_episode_reward / len(episode_rewards)

        logger.log_evaluation_metrics(episode, step, total_episode_reward)

        print(
            f"Evaluation Episode {episode + 1}: Total Reward = {total_episode_reward}, Avrg. Reward = {average_reward}, Steps = {step}, Epsilon = {agent.DQN.epsilon:.4f}")


    agent.DQN.epsilon = original_epsilon  # Restore original epsilon
    avg_reward = sum(episode_rewards) / num_episodes
    print(f"Average Evaluation Reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward

def adjust_max_steps(average_reward, threshold, last_steps, max_steps_increment):

    if average_reward > threshold:
        last_steps += max_steps_increment
        print(f"Increasing max_steps to {last_steps}")

    return last_steps

if __name__ == "__main__":
    main()

