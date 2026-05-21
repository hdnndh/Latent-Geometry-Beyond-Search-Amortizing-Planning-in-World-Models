#!/usr/bin/env python3
"""Evaluate MPPI / iCEM / GradientSolver on the LeWM cost model.

Same eval protocol as eval_idm.py CEM branch, only the solver differs.
All solvers use swm class defaults (GradientSolver gets n_steps=30).

Usage:
    python eval_othersolvers.py --solver mppi --dataset tworoom --num-eval 200
    python eval_othersolvers.py --solver icem --dataset pusht --num-eval 50
    python eval_othersolvers.py --solver gradient --dataset tworoom --num-eval 50
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import stable_worldmodel as swm

from eval_idm import (
    DATASET_CONFIGS,
    get_img_transform,
    build_normalizers,
    sample_eval_episodes,
)
from idm.dataset import load_lewm_model


# ---------------------------------------------------------------------------
# Solver factory
# ---------------------------------------------------------------------------

SOLVER_REGISTRY = {
    "mppi": "MPPISolver",
    "icem": "ICEMSolver",
    "gradient": "GradientSolver",
}


def _make_solver(
    name: str,
    cost_model,
    device,
    seed: int,
    *,
    grad_n_steps: int = 30,
):
    """Instantiate a solver by name using class defaults for all shared params.

    Shared defaults for CEM / MPPI / iCEM:
        num_samples=300, n_steps=30, topk=30, var_scale=1.0
    Solver-specific defaults:
        MPPISolver:  temperature=0.5
        ICEMSolver:  noise_beta=2.0, alpha=0.1, n_elite_keep=5, return_mean=True

    GradientSolver uses pure swm class defaults except:
      - `n_steps` is REQUIRED (no default) — we pass 30 to match the
        iteration count of the other solvers.
      - `num_samples=2` instead of default 1 — LeWM's criterion has a
        shape ambiguity when num_samples=1 (goal_emb loses a dimension
        during encoding). num_samples=2 is the minimum fix. Also gives
        gradient-based planning a second restart, which is standard.
      - Everything else uses swm's defaults exactly:
            var_scale=1, action_noise=0.0, optimizer_cls=SGD,
            optimizer_kwargs=None
    """
    class_name = SOLVER_REGISTRY.get(name)
    if class_name is None:
        raise ValueError(f"Unknown solver '{name}'. Choose from: {list(SOLVER_REGISTRY)}")

    cls = getattr(swm.solver, class_name, None)
    if cls is None:
        available = [x for x in dir(swm.solver) if "Solver" in x]
        raise ImportError(
            f"swm.solver.{class_name} not found. "
            f"Available: {available}. "
            f"Try: pip install --upgrade stable-worldmodel"
        )

    if name == "gradient":
        # n_steps=30 (required, no default), num_samples=2 (avoids shape bug with 1)
        return cls(
            model=cost_model,
            device=device,
            seed=seed,
            n_steps=grad_n_steps,
            num_samples=2,
        )

    # CEM / MPPI / iCEM — pure class defaults
    return cls(model=cost_model, device=device, seed=seed)


# ---------------------------------------------------------------------------
# Main eval function
# ---------------------------------------------------------------------------

def run_solver_eval(
    dataset: str,
    solver_name: str,
    num_eval: int = 200,
    eval_budget: int = 50,
    goal_offset: int = 25,
    seed: int = 42,
    train_split: float = 1.0,
    split_seed: int = 42,
    device_str: str = "cuda:0",
    grad_n_steps: int = 30,
):
    """Run a sampling-based (or gradient-based) planner eval.

    The pipeline mirrors eval_idm.py's CEM branch:
      1. Load AutoCostModel
      2. Create solver (MPPI, iCEM, or GradientSolver)
      3. Same PlanConfig (horizon=5, receding_horizon=5, action_block=5)
      4. Same warmup (encoder JIT + page cache)
      5. Same timed eval via world.evaluate(dataset=...)
    """
    solver_label = {"mppi": "MPPI", "icem": "iCEM", "gradient": "GRAD"}[solver_name]

    dcfg = DATASET_CONFIGS[dataset]
    device = torch.device(device_str)
    data_dir = os.environ.get("STABLEWM_HOME", "./stable-wm-data")
    ckpt_dir = os.path.join(data_dir, "checkpoints", dataset, "lewm")

    # --- Cost model (same as eval_idm.py CEM branch) ---
    ckpt_obj = os.path.join(ckpt_dir, "lewm_object.ckpt")
    if not os.path.exists(ckpt_obj) and os.path.exists(
        os.path.join(ckpt_dir, "config.json")
    ):
        print(f"Converting checkpoint to _object.ckpt for {solver_label}...")
        _model = load_lewm_model(ckpt_dir, "cpu")
        torch.save(_model, ckpt_obj)
        del _model

    cost_model = swm.policy.AutoCostModel(f"{dataset}/lewm")
    cost_model = cost_model.to(device).eval()
    cost_model.requires_grad_(False)
    cost_model.interpolate_pos_encoding = True

    # Fix: goal_emb shape mismatch when num_samples > 1.
    import types
    import torch.nn.functional as F

    def _patched_criterion(self, info_dict: dict):
        pred_emb = info_dict['predicted_emb']  # (B, S, T, dim) — 4D
        goal_emb = info_dict['goal_emb']       # might be (B, 1, dim) — 3D

        # Insert missing dims so goal_emb matches pred_emb's ndim
        while goal_emb.ndim < pred_emb.ndim:
            goal_emb = goal_emb.unsqueeze(-2)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction='none',
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    cost_model.criterion = types.MethodType(_patched_criterion, cost_model)

    # --- Solver ---
    solver = _make_solver(
        solver_name, cost_model, device, seed,
        grad_n_steps=grad_n_steps,
    )

    # --- PlanConfig (matches LeWM's eval config exactly) ---
    config = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5)

    # --- Dataset + transforms ---
    dataset_obj = swm.data.HDF5Dataset(
        dcfg.dataset_name, keys_to_cache=dcfg.cache_keys
    )
    img_tf = get_img_transform(dcfg.img_size)
    transform = {"pixels": img_tf, "goal": img_tf}
    process = build_normalizers(dataset_obj, dcfg.cache_keys)

    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process, transform=transform,
    )

    # --- Sample eval episodes ---
    eval_episodes, eval_start_idx = sample_eval_episodes(
        dataset_obj, num_eval, goal_offset, seed,
        train_split=train_split, split_seed=split_seed,
    )

    n_eval = len(eval_episodes)
    world = swm.World(
        env_name=dcfg.env,
        num_envs=n_eval,
        image_shape=(dcfg.img_size, dcfg.img_size),
        max_episode_steps=2 * max(eval_budget, goal_offset) + 5,
        **(dcfg.world_kwargs or {}),
    )
    world.set_policy(policy)

    # Fix: action bounds shape mismatch with action_block > 1.
    if solver_name in ("icem", "gradient") and hasattr(solver, "_action_low"):
        ab = config.action_block
        if solver._action_low.shape[0] != solver.action_dim:
            solver._action_low = solver._action_low.repeat(ab)
            solver._action_high = solver._action_high.repeat(ab)
            print(f"  (patched {solver_label} action bounds: "
                  f"{solver._action_low.shape[0]}D for action_block={ab})")

    # --- Warmup (NOT timed) ---
    print(f"  Warming up {solver_label} (encoder + page cache)...")

    # Encoder warmup
    _dummy = torch.randn(
        min(n_eval, 8), 3, dcfg.img_size, dcfg.img_size, device=device
    )
    with torch.no_grad():
        for _ in range(3):
            cost_model.encoder(_dummy, interpolate_pos_encoding=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    del _dummy

    # Page-cache warmup
    _ep_arr = np.array(eval_episodes.tolist())
    _start_arr = np.array(eval_start_idx.tolist())
    _end_arr = _start_arr + goal_offset + 1
    _ = dataset_obj.load_chunk(_ep_arr, _start_arr, _end_arr)
    del _

    # Per-step timing wrapper
    plan_times: list[float] = []
    _orig_get_action = policy.get_action

    def _timed_get_action(info_dict, **kwargs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        _t = time.perf_counter()
        _result = _orig_get_action(info_dict, **kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        plan_times.append(time.perf_counter() - _t)
        return _result

    policy.get_action = _timed_get_action

    # --- Timed eval ---
    t0 = time.time()
    metrics = world.evaluate(
        dataset=dataset_obj,
        start_steps=eval_start_idx.tolist(),
        goal_offset=goal_offset,
        eval_budget=eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=dcfg.callables,
    )
    elapsed = time.time() - t0
    ms_per_ep = 1000 * elapsed / n_eval
    plan_ms = (
        1000 * sum(plan_times) / len(plan_times) if plan_times else 0.0
    )

    sr = metrics.get("success_rate", "?")


    print(f"\n{solver_label} metrics: {metrics}")
    print(
        f"{solver_label} wall-clock: {elapsed:.1f}s total, "
        f"{ms_per_ep:.0f} ms/episode"
    )
    print(
        f"{solver_label} pure-compute: {plan_ms:.1f} ms/plan-call "
        f"(over {len(plan_times)} calls)"
    )

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  solver:         {solver_label}")
    print(f"  dataset:        {dataset}")
    print(f"  num_eval:       {n_eval}")
    print(f"  eval_budget:    {eval_budget}")
    print(f"  goal_offset:    {goal_offset}")
    print(f"  seed:           {seed}")
    if solver_name == "gradient":
        print(f"  grad_n_steps:   {grad_n_steps}")
        print(f"  grad_defaults:  num_samples=2, optimizer=SGD (swm defaults)")
    print(f"  success_rate:   {sr}")
    print(f"  ms_per_episode: {ms_per_ep:.0f}")
    print(f"  ms_per_plan:    {plan_ms:.1f}")
    print(f"  total_time:     {elapsed:.1f}s")

    return metrics, ms_per_ep, plan_ms


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MPPI / iCEM / GradientSolver on the LeWM cost model"
    )
    parser.add_argument(
        "--solver",
        required=True,
        choices=list(SOLVER_REGISTRY),
        help="Which solver to evaluate",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
    )
    parser.add_argument("--num-eval", type=int, default=200)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-split", type=float, default=1.0,
                        help="If < 1.0, eval only on held-out episodes")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")

    # Gradient-specific: only n_steps is exposed (required by swm).
    # All other GradientSolver params use swm class defaults

    parser.add_argument(
        "--grad-n-steps", type=int, default=30,
        help="Gradient iterations (default: 30, matches other solvers' n_steps)",
    )

    args = parser.parse_args()

    run_solver_eval(
        dataset=args.dataset,
        solver_name=args.solver,
        num_eval=args.num_eval,
        eval_budget=args.eval_budget,
        goal_offset=args.goal_offset,
        seed=args.seed,
        train_split=args.train_split,
        split_seed=args.split_seed,
        device_str=args.device,
        grad_n_steps=args.grad_n_steps,
    )


if __name__ == "__main__":
    main()
