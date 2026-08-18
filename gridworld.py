"""
Lab Assignment 3: GridWorld - Policy Evaluation and Value Iteration
=====================================================================

Environment layout (2 rows x 3 columns, 6 cells, 5 non-terminal states + 1 goal):

    S1   S2   S3
    S4   S5   GOAL

Actions available in every state: UP, DOWN, LEFT, RIGHT
- Moving off the grid boundary leaves the agent in the same cell (a "bump").
- Every move costs a reward of -1 (including bumps).
- Reaching GOAL gives a reward of +10 and ends the episode.
- Discount factor gamma = 0.9
"""

import random
import copy

random.seed(42)

# ---------------------------------------------------------------------------
# 1. Environment definition
# ---------------------------------------------------------------------------

ROWS, COLS = 2, 3
GOAL = (1, 2)
GAMMA = 0.9
STEP_REWARD = -1
GOAL_REWARD = 10

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_DELTA = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

# Map grid coordinates <-> state labels S1..S5 (GOAL is terminal, not in the value table)
COORD_TO_LABEL = {
    (0, 0): "S1", (0, 1): "S2", (0, 2): "S3",
    (1, 0): "S4", (1, 1): "S5", (1, 2): "GOAL",
}
LABEL_TO_COORD = {v: k for k, v in COORD_TO_LABEL.items()}
NON_TERMINAL_STATES = [(r, c) for r in range(ROWS) for c in range(COLS) if (r, c) != GOAL]


def step(state, action):
    """Return (next_state, reward, done) for taking `action` in `state`."""
    if state == GOAL:
        return state, 0, True
    dr, dc = ACTION_DELTA[action]
    nr, nc = state[0] + dr, state[1] + dc
    if 0 <= nr < ROWS and 0 <= nc < COLS:
        next_state = (nr, nc)
    else:
        next_state = state  # bump into the wall, stay in place
    reward = GOAL_REWARD if next_state == GOAL else STEP_REWARD
    done = next_state == GOAL
    return next_state, reward, done


# ---------------------------------------------------------------------------
# 2. A fixed "given" policy to evaluate (heuristic: prefer RIGHT, else DOWN)
# ---------------------------------------------------------------------------

def given_policy(state):
    """Deterministic heuristic policy: move RIGHT if possible progress toward
    goal column, else move DOWN. Used for Task 2 (Policy Evaluation)."""
    r, c = state
    if c < COLS - 1:
        return "RIGHT"
    else:
        return "DOWN"


# ---------------------------------------------------------------------------
# 3. Policy Evaluation (iterative, for the given deterministic policy)
# ---------------------------------------------------------------------------

def policy_evaluation(policy_fn, theta=1e-4, max_iterations=100, track_iterations=5):
    V = {s: 0.0 for s in NON_TERMINAL_STATES}
    V[GOAL] = 0.0
    history = []  # (iteration, max_delta, converged?)

    for it in range(1, max_iterations + 1):
        delta = 0.0
        new_V = copy.deepcopy(V)
        for s in NON_TERMINAL_STATES:
            a = policy_fn(s)
            next_s, r, done = step(s, a)
            new_V[s] = r + GAMMA * V[next_s]
            delta = max(delta, abs(new_V[s] - V[s]))
        V = new_V
        converged = delta < theta
        if it <= track_iterations or converged:
            history.append((it, delta, converged))
        if converged:
            break
    return V, history


# ---------------------------------------------------------------------------
# 4. Value Iteration
# ---------------------------------------------------------------------------

def value_iteration(theta=1e-4, max_iterations=1000):
    V = {s: 0.0 for s in NON_TERMINAL_STATES}
    V[GOAL] = 0.0

    for it in range(1, max_iterations + 1):
        delta = 0.0
        new_V = copy.deepcopy(V)
        for s in NON_TERMINAL_STATES:
            action_values = []
            for a in ACTIONS:
                next_s, r, done = step(s, a)
                action_values.append(r + GAMMA * V[next_s])
            new_V[s] = max(action_values)
            delta = max(delta, abs(new_V[s] - V[s]))
        V = new_V
        if delta < theta:
            break

    # Derive greedy optimal policy from V*
    optimal_policy = {}
    for s in NON_TERMINAL_STATES:
        best_a, best_val = None, float("-inf")
        for a in ACTIONS:
            next_s, r, done = step(s, a)
            val = r + GAMMA * V[next_s]
            if val > best_val:
                best_val = val
                best_a = a
        optimal_policy[s] = best_a
    return V, optimal_policy, it


# ---------------------------------------------------------------------------
# 5. Policy execution / rollout utilities
# ---------------------------------------------------------------------------

def run_deterministic_policy(policy_fn, start=(0, 0), max_steps=20):
    state = start
    path = [COORD_TO_LABEL[state]]
    total_reward = 0
    for _ in range(max_steps):
        if state == GOAL:
            break
        a = policy_fn(state)
        next_state, r, done = step(state, a)
        total_reward += r
        state = next_state
        path.append(COORD_TO_LABEL[state])
        if done:
            break
    reached = state == GOAL
    return path, len(path) - 1, total_reward, reached


def run_random_policy(start=(0, 0), max_steps=20, seed=42):
    rng = random.Random(seed)
    state = start
    path = [COORD_TO_LABEL[state]]
    total_reward = 0
    for _ in range(max_steps):
        if state == GOAL:
            break
        a = rng.choice(ACTIONS)
        next_state, r, done = step(state, a)
        total_reward += r
        state = next_state
        path.append(COORD_TO_LABEL[state])
        if done:
            break
    reached = state == GOAL
    return path, len(path) - 1, total_reward, reached


# ---------------------------------------------------------------------------
# 6. Run everything and print results
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("TASK 2: POLICY EVALUATION (given heuristic policy)")
    print("=" * 70)
    V_eval, history = policy_evaluation(given_policy)
    print(f"{'Iteration':<12}{'Max Change (Delta)':<22}{'Convergence Status'}")
    for it, delta, converged in history:
        status = "Converged" if converged else "Not yet converged"
        print(f"{it:<12}{delta:<22.6f}{status}")
    print("\nState values under the evaluated policy:")
    for s in NON_TERMINAL_STATES:
        print(f"  {COORD_TO_LABEL[s]}: {V_eval[s]:.3f}   (action: {given_policy(s)})")

    print("\n" + "=" * 70)
    print("TASK 3: VALUE ITERATION")
    print("=" * 70)
    V_star, pi_star, iters_used = value_iteration()
    print(f"Converged in {iters_used} iterations\n")
    print(f"{'State':<8}{'Optimal Action':<18}{'State Value'}")
    for s in NON_TERMINAL_STATES:
        print(f"{COORD_TO_LABEL[s]:<8}{pi_star[s]:<18}{V_star[s]:.3f}")

    print("\n" + "=" * 70)
    print("TASK 4: OPTIMAL PATH ANALYSIS")
    print("=" * 70)

    rand_path, rand_len, rand_reward, rand_reached = run_random_policy()
    eval_path, eval_len, eval_reward, eval_reached = run_deterministic_policy(given_policy)
    opt_path, opt_len, opt_reward, opt_reached = run_deterministic_policy(lambda s: pi_star[s])

    print(f"{'Policy Type':<18}{'Path Length':<14}{'Total Reward':<15}{'Goal Reached'}")
    print(f"{'Random Policy':<18}{rand_len:<14}{rand_reward:<15}{'Yes' if rand_reached else 'No'}")
    print(f"{'Evaluated Policy':<18}{eval_len:<14}{eval_reward:<15}{'Yes' if eval_reached else 'No'}")
    print(f"{'Optimal Policy':<18}{opt_len:<14}{opt_reward:<15}{'Yes' if opt_reached else 'No'}")

    print("\nSample paths:")
    print("  Random policy   :", " -> ".join(rand_path))
    print("  Evaluated policy:", " -> ".join(eval_path))
    print("  Optimal policy  :", " -> ".join(opt_path))
