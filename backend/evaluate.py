import os

from envs.hetnet_env import DynamicHetNetEnv
from models.dqn import DQNAgent
from models.baselines import StrictPriorityScheduler, RoundRobinScheduler, WeightedFairQueuingScheduler

CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "drl_hetnet_model.pth")
POLICY_NAMES = ["DRL Policy", "Strict Priority", "Round Robin", "Weighted Fair Queuing"]


def load_agent(path=CHECKPOINT_PATH):
    """Returns (agent, loaded). An untrained agent is returned if no checkpoint exists."""
    agent = DQNAgent()
    if not os.path.exists(path):
        return agent, False
    agent.load(path)
    return agent, True


def make_policies(drl_agent):
    """Builds a fresh set of policy functions keyed by name.

    Baselines like Round Robin keep internal state, so each run gets new instances.
    """
    sp, rr, wfq = StrictPriorityScheduler(), RoundRobinScheduler(), WeightedFairQueuingScheduler()
    return {
        "DRL Policy": lambda s: drl_agent.select_action(s, eval_mode=True),
        "Strict Priority": sp.select_action,
        "Round Robin": rr.select_action,
        "Weighted Fair Queuing": wfq.select_action,
    }


def run_episode(env, policy_fn, seed=None):
    """Plays one full episode and returns its aggregate QoS metrics."""
    state, _ = env.reset(seed=seed)
    totals = {"reward": 0.0, "served": 0, "drops": 0, "sla_violations": 0}

    while True:
        state, reward, terminated, truncated, info = env.step(policy_fn(state))
        totals["reward"] += reward
        totals["served"] += info["served_count"]
        totals["drops"] += sum(info["drops"])
        totals["sla_violations"] += sum(info["sla_violations"])
        if terminated or truncated:
            return totals


def benchmark(drl_agent, episode_length=200, seed=0, mmwave_blocked=False):
    """Runs every policy against the same seeded traffic so the comparison is fair."""
    env = DynamicHetNetEnv(episode_length=episode_length)
    env.set_mmwave_blockage(mmwave_blocked)
    return {
        name: run_episode(env, policy_fn, seed=seed)
        for name, policy_fn in make_policies(drl_agent).items()
    }


def print_table(title, results):
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    print(f"{'Policy':<22} | {'Total Reward':<12} | {'Packets Served':<15} | {'Packet Drops':<13} | {'SLA Violations':<14}")
    print("-" * 80)
    for name, r in results.items():
        print(f"{name:<22} | {r['reward']:12.2f} | {r['served']:15d} | {r['drops']:13d} | {r['sla_violations']:14d}")
    print("=" * 80)


def evaluate_all(checkpoint_path=CHECKPOINT_PATH, seeds=(0,)):
    drl_agent, loaded = load_agent(checkpoint_path)
    if loaded:
        print(f"Loaded trained model from {checkpoint_path}")
    else:
        print("Warning: Model checkpoint not found. Using untrained agent.")

    all_results = {}
    for blocked in (False, True):
        # Sum over several traffic seeds so one lucky episode doesn't decide the ranking.
        totals = {}
        for seed in seeds:
            for name, r in benchmark(drl_agent, episode_length=300, seed=seed, mmwave_blocked=blocked).items():
                acc = totals.setdefault(name, dict.fromkeys(r, 0))
                for key, value in r.items():
                    acc[key] += value
        scenario = "MMWAVE BLOCKED" if blocked else "NORMAL CONDITIONS"
        print_table(f"SCHEDULER BENCHMARK: {scenario} ({len(seeds)} seed(s))", totals)
        all_results["blocked" if blocked else "normal"] = totals
    print()
    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark the DQN against the baseline schedulers.")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--seeds", type=int, default=1, help="number of traffic seeds to sum over")
    args = parser.parse_args()
    evaluate_all(args.checkpoint, seeds=range(args.seeds))
