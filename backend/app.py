import os
import random
import threading

from flask import Flask, request, jsonify, send_from_directory

from envs.hetnet_env import DynamicHetNetEnv, DEFAULT_ARRIVAL_RATES
from evaluate import POLICY_NAMES, load_agent, make_policies, benchmark

# Arrival rates are Poisson means in packets per slot. The upper bound keeps a
# single request from generating millions of packets.
MAX_ARRIVAL_RATE = 50.0

# Long enough that one lucky stretch of traffic can't flip the ranking.
BENCHMARK_SLOTS = 2000

# Point Flask to the frontend directory
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
app = Flask(__name__, static_folder=frontend_dir, static_url_path="")

# Global simulation state. Flask serves requests on multiple threads, so every
# read or write of the env goes through sim_lock.
sim_lock = threading.Lock()
env = DynamicHetNetEnv()
drl_agent, model_loaded = load_agent()
print("Loaded DRL model checkpoint." if model_loaded else "No checkpoint found. DRL policy is untrained.")

policies = make_policies(drl_agent)
current_state, _ = env.reset()


def parse_arrival_rates(raw):
    if raw is None:
        return list(DEFAULT_ARRIVAL_RATES)
    if not isinstance(raw, list) or len(raw) != env.num_classes:
        raise ValueError(f"arrival_rates must be a list of {env.num_classes} numbers")
    try:
        rates = [float(r) for r in raw]
    except (TypeError, ValueError):
        raise ValueError("arrival_rates must contain only numbers")
    if any(not 0.0 <= r <= MAX_ARRIVAL_RATE for r in rates):
        raise ValueError(f"arrival_rates must be between 0 and {MAX_ARRIVAL_RATE}")
    return rates


@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/reset", methods=["POST"])
def reset_simulation():
    global current_state
    with sim_lock:
        env.set_mmwave_blockage(False)
        current_state, _ = env.reset()
        return jsonify({"status": "reset", "state": current_state.tolist(), "mmwave_blocked": False})


@app.route("/api/toggle-blockage", methods=["POST"])
def toggle_blockage():
    data = request.get_json(silent=True) or {}
    with sim_lock:
        env.set_mmwave_blockage(bool(data.get("blocked", False)))
        return jsonify({"status": "updated", "mmwave_blocked": env.mmwave_blocked})


@app.route("/api/step", methods=["POST"])
def step_simulation():
    global current_state
    data = request.get_json(silent=True) or {}

    policy_name = data.get("policy", "DRL Policy")
    if policy_name not in policies:
        return jsonify({"error": f"Unknown policy '{policy_name}'", "policies": POLICY_NAMES}), 400
    try:
        arrival_rates = parse_arrival_rates(data.get("arrival_rates"))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    with sim_lock:
        action = policies[policy_name](current_state)
        next_state, reward, terminated, truncated, info = env.step(action, arrival_rates=arrival_rates)
        episode_over = terminated or truncated
        # Per-packet waiting time, oldest first, so the UI can show which packets are past their SLA.
        packet_ages = [[env.current_step - t for t in q] for q in env.queues]
        # The response reports the state the action produced; the next request
        # starts from a fresh episode if this one just ended.
        current_state = env.reset()[0] if episode_over else next_state
        mmwave_blocked = env.mmwave_blocked

    return jsonify({
        "state": next_state.tolist(),
        "reward": float(reward),
        "done": episode_over,
        "info": info,
        "queue_lengths": next_state[0:3].tolist(),
        "hol_delays": next_state[3:6].tolist(),
        "arrival_rates": next_state[6:9].tolist(),
        "link_capacities": next_state[9:12].tolist(),
        "mmwave_blocked": mmwave_blocked,
        "packet_ages": packet_ages,
        "sla_targets": env.sla_targets,
        "action_taken": {
            "queue_idx": info["queue_idx"],
            "link_idx": info["link_idx"],
        },
    })


@app.route("/api/benchmark", methods=["GET"])
def get_benchmark():
    # Runs on its own env and fresh baseline instances, so it never touches the live simulation.
    # Each run draws new traffic, but every scheduler in that run sees the same seed.
    # Pass ?seed=N to repeat a specific run.
    seed = request.args.get("seed", type=int)
    if seed is None:
        seed = random.randrange(100_000)
    scenarios = {}
    for key, blocked in (("results", False), ("results_blocked", True)):
        results = benchmark(drl_agent, episode_length=BENCHMARK_SLOTS, seed=seed, mmwave_blocked=blocked)
        for r in results.values():
            r["reward"] = round(r["reward"], 2)
        scenarios[key] = results
    return jsonify({"model_loaded": model_loaded, "seed": seed, **scenarios})


if __name__ == "__main__":
    # The Werkzeug debugger allows arbitrary code execution, so it is opt-in
    # and the server only listens on this machine by default.
    debug = os.environ.get("FLASK_DEBUG") == "1"
    host = os.environ.get("HOST", "127.0.0.1")
    app.run(host=host, port=5000, debug=debug)
