import gymnasium as gym
import tensorflow as tf
from datetime import datetime

from Agent import Agent

max_steps = 100

action_order = ["steering", "gas", "breaking"]

def main():
    env = gym.make("CarRacing-v2", render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/agent_performance/" + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)
    run(env, summary_writer)


def setup_agents():
    pass


def run(env, log_writer):

    agent = Agent()

    # run the game loop
    step = 0
    # list of agents that are still running, by index in agents list
    done = False
    observation = env.reset()

    while not done and step < max_steps:
        env.render()

        action = agent.get_action(observation)

        observation, reward, _, all_done, _ = env.step(action)

        log_stats(reward, step, action, log_writer)

        step += 1

    env.close()

def log_stats(reward, step, actions, writer):
    #if step % self.settings.get_setting("storing_round_interval") == 0:
        with writer.as_default():
            tf.summary.scalar("reward", reward, step)
            for action, action_name in zip(actions, action_order):
                tf.summary.scalar(action_name,action , step)

    # if step % self.settings.get_setting("image_logging_round_interval") == 0:
    #     game_state_image = capture_game_state_as_image()
    #     tensor_img = tf.convert_to_tensor(game_state_image, dtype=tf.uint8)
    #     tensor_img = tf.expand_dims(tensor_img, 0)  # Add the batch dimension
    #     with self.summary_writer.as_default():
    #         tf.summary.image("Step: " + str(step), tensor_img, step)

if __name__ == "__main__":
    main()
