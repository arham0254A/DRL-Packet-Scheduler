import argparse
import os

import numpy as np

from envs.hetnet_env import DynamicHetNetEnv
from models.dqn import DQNAgent
from evaluate import CHECKPOINT_PATH

# Share of training episodes that include a forced mmWave blockage. Random
# per-slot blockage alone almost never lasts more than a slot or two, so without
# these the agent never learns to move traffic off a link that stays dead.
BLOCKED_EPISODE_SHARE = 0.4


def sample_blockage_window(rng, episode_length):
    """Returns (start, end) slots of a forced blockage, or None for a normal episode.

    Half of the blocked episodes are blocked from the first slot; the rest get a
    window that starts mid-episode, so the agent also sees the link going down
    and coming back.
    """
    if rng.random() >= BLOCKED_EPISODE_SHARE:
        return None
    if rng.random() < 0.5:
        return 0, episode_length
    start = int(rng.integers(0, episode_length - 30))
    return start, start + int(rng.integers(30, 120))


def run_training(episodes=300, batch_size=64, checkpoint_path=CHECKPOINT_PATH, seed=None):
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    env = DynamicHetNetEnv()
    agent = DQNAgent()
    rng = np.random.default_rng(seed)

    print("=" * 60)
    print("Starting Deep Reinforcement Learning HetNet Scheduler Training")
    print("=" * 60)

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        window = sample_blockage_window(rng, env.episode_length)
        total_reward = 0.0
        total_loss = 0.0
        steps = 0

        while True:
            env.set_mmwave_blockage(window is not None and window[0] <= steps < window[1])
            action = agent.select_action(state, eval_mode=False)
            next_state, reward, terminated, truncated, info = env.step(action)

            # Only a true terminal state stops bootstrapping. Hitting the episode
            # time limit (truncated) still has future value.
            agent.memory.push(state, action, reward, next_state, float(terminated))
            loss = agent.train_step(batch_size)

            state = next_state
            total_reward += reward
            total_loss += loss
            steps += 1

            if terminated or truncated:
                break

        agent.decay_epsilon()

        if ep % 10 == 0:
            agent.update_target_network()
            avg_loss = total_loss / max(1, steps)
            print(f"Episode {ep:03d}/{episodes:03d} | Total Reward: {total_reward:8.2f} | "
                  f"Avg Loss: {avg_loss:6.4f} | Epsilon: {agent.epsilon:.3f}", flush=True)

    env.set_mmwave_blockage(False)
    agent.save(checkpoint_path)
    print("=" * 60)
    print(f"Training Complete! Checkpoint saved to: {checkpoint_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the DQN packet scheduler.")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--out", default=CHECKPOINT_PATH, help="where to save the checkpoint")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    run_training(episodes=args.episodes, checkpoint_path=args.out, seed=args.seed)
