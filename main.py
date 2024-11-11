import gymnasium as gym
import tensorflow as tf
from datetime import datetime

from Agent import Agent

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
    summary_writer = tf.summary.create_file_writer(log_dir)
    model_path = "models/" + current_time + "_model"
    run(env, summary_writer, model_path)

def run(env, log_writer, model_path):
    agent = Agent()
    steps_per_episode = 100
    total_steps_taken = 0

    for episode in range(num_episodes):


        # run the game loop
        step = 0
        # list of agents that are still running, by index in agents list
        done = False
        info = env.reset()
        old_observation = info[0]
        new_observation = None
        episode_rewards = []
        episode_loss = []

        while not done and step < steps_per_episode:
            if env.render_mode == "human":
                env.render()

            # if total_reward < cut_off_reward:
            #     continue

            action = agent.get_action(old_observation)

            new_observation, reward, _, all_done, _ = env.step(action)
            agent.store_transition(old_observation, action, reward, new_observation, all_done)

            loss = agent.train()
            episode_loss.append(loss)

            episode_rewards.append(reward)
            old_observation = new_observation
            step += 1

        total_steps_taken += step
        total_episode_reward = sum(episode_rewards)
        average_reward = total_episode_reward / len(episode_rewards)

        log_stats(total_episode_reward, average_reward, episode, episode_loss, log_writer, agent)
        print(
            f"Episode {episode + 1}: Total Reward = {total_episode_reward}, Avrg. Reward = {average_reward}, Steps = {step}, Epsilon = {agent.DQN.epsilon:.4f}")

        steps_per_episode = adjust_max_steps(average_reward, performance_threshold, max_steps_per_episode, episode_length_increment)

    env.close()
    agent.save_model(model_path)
    print("TRAINING COMPLETE")
    print(f"Total steps taken: {total_steps_taken}")

    # TODO: Evaluate the agent

def log_stats(total_reward, average_reward, episode, loss, writer, agent):
    #if step % self.settings.get_setting("storing_round_interval") == 0:
        with writer.as_default():
            # tf.summary.scalar("reward", reward, step)
            # for action, action_name in zip(actions, action_order):
            #     tf.summary.scalar(action_name,action , step)

                #tf.summary.scalar('Loss/train', loss, episode)
                tf.summary.scalar('total reward/episode', total_reward, episode)
                tf.summary.scalar('average reward/episode', average_reward, episode)
                tf.summary.scalar('Epsilon', agent.DQN.epsilon, episode)

    # if step % self.settings.get_setting("image_logging_round_interval") == 0:
    #     game_state_image = capture_game_state_as_image()
    #     tensor_img = tf.convert_to_tensor(game_state_image, dtype=tf.uint8)
    #     tensor_img = tf.expand_dims(tensor_img, 0)  # Add the batch dimension
    #     with self.summary_writer.as_default():
    #         tf.summary.image("Step: " + str(step), tensor_img, step)

def evaluate_agent(agent, env, action_mapping, num_episodes=10):
    total_rewards = []
    original_epsilon = agent.DQN.epsilon
    agent.DQN.epsilon = 0.0  # Disable exploration

    for episode in range(eval_episodes):
        state = env.reset()
        done = False
        episode_rewards = []
        obs = state[0]

        while not done:
            action = agent.get_action(obs)  # Select action without exploration

            next_state, reward, done, info = env.step(action)
            obs = next_state
            episode_rewards.append(reward)

        total_reward = sum(episode_rewards)
        average_reward = total_reward / len(episode_rewards)

        total_rewards.append(total_reward)
        print(f"Evaluation Episode {episode+1}: Total Reward = {total_reward}")

    agent.DQN.epsilon = original_epsilon  # Restore original epsilon
    avg_reward = sum(total_rewards) / num_episodes
    print(f"Average Evaluation Reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward

def adjust_max_steps(average_reward, threshold, max_steps, max_steps_increment):

    if average_reward > threshold:
        max_steps += max_steps_increment
        print(f"Increasing max_steps to {max_steps}")

    return max_steps

if __name__ == "__main__":
    main()

