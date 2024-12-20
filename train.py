import time

import gymnasium as gym


def train(env, agent, logger, hyperparameters):
    steps_per_episode = hyperparameters.get("start_episode_length")
    training_start_time = time.time()
    max_total_steps = hyperparameters.get("max_total_steps")
    total_steps_taken = agent.DQN.steps_done  # 0 if not continuing a training

    if total_steps_taken >= max_total_steps:
        print("Agent has already trained for the maximum number of steps")
        return

    episode = logger.total_episodes  # 0 if not continuing a training
    print("STARTING TRAINING")
    print("planning to take {} steps".format(max_total_steps - total_steps_taken))
    while total_steps_taken < max_total_steps:
        episode += 1

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

            action = agent.get_action(old_observation)

            new_observation, reward, _, all_done, _ = env.step(action)
            agent.store_transition(
                old_observation, action, reward, new_observation, all_done
            )

            agent.train()

            episode_rewards.append(reward)
            # logger.log_action(total_steps_taken, action)

            old_observation = new_observation
            step += 1
            total_steps_taken += 1

        total_episode_reward = sum(episode_rewards)
        average_reward = total_episode_reward / len(episode_rewards)
        # agent.save_checkpoint(total_steps_taken, max_total_steps, logger.checkpoint_path)
        logger.log_episode_metrics(
            total_steps_taken,
            step,
            total_episode_reward,
            agent.DQN.epsilon,
            len(agent.DQN.memory),
        )

        print(
            f"Episode {episode + 1}: Total Reward = {total_episode_reward}, Avrg. Reward = {average_reward}, Steps = {step}, Epsilon = {agent.DQN.epsilon:.4f}"
        )

        steps_per_episode = adjust_max_steps(
            average_reward,
            hyperparameters.get("performance_threshold"),
            step,
            hyperparameters.get("episode_length_increment"),
        )

    training_end_time = time.time()

    total_training_time = training_end_time - training_start_time
    print("TRAINING COMPLETE")
    print(f"Total steps taken: {total_steps_taken}")
    print(f"Average steps per Episode: {total_steps_taken / episode}")
    print(f"Training time: {total_training_time:.2f} seconds")
    print(f"steps/seconds: {total_steps_taken / total_training_time:.2f}")


def adjust_max_steps(average_reward, threshold, last_steps, max_steps_increment):
    if average_reward > threshold:
        last_steps += max_steps_increment
        print(f"Increasing max_steps to {last_steps}")

    return last_steps


def evaluate_agent(agent, logger, num_episodes=10, eval_steps=200, render=False):
    original_epsilon = agent.DQN.epsilon
    agent.DQN.epsilon = 0.0  # Disable exploration
    render_episode = False

    env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
    )

    print("STARTING EVALUATION")
    for episode in range(num_episodes):
        if render and episode == num_episodes - 1:
            render_episode = True

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
            if render_episode:
                # log the image
                rgb_array = env.render()
                logger.log_image(rgb_array, step)

            action = agent.get_action(old_observation)

            new_observation, reward, _, all_done, _ = env.step(action)

            episode_rewards.append(reward)

            old_observation = new_observation
            step += 1

        total_steps_taken += step
        total_episode_reward = sum(episode_rewards)
        average_reward = total_episode_reward / len(episode_rewards)

        logger.log_evaluation_metrics(step, total_episode_reward)

        print(
            f"Evaluation Episode {episode + 1}: Total Reward = {total_episode_reward}, Avrg. Reward = {average_reward}, Steps = {step}, Epsilon = {agent.DQN.epsilon:.4f}"
        )

    if render_episode:
        logger.create_gif()
    agent.DQN.epsilon = original_epsilon  # Restore original epsilon
    avg_reward = sum(episode_rewards) / num_episodes
    print(f"Average Evaluation Reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward
