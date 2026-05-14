#!/usr/bin/env python3
# script: run_gemf_recovery_simulations.py

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCENARIOS = {
    "shared_only": {"Ls": 0, "Lf": 0, "Le": 0},
    "shared_specific": {"Ls": 1, "Lf": 1, "Le": 1},
    "low_snr_specific": {"Ls": 1, "Lf": 1, "Le": 1},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GEMF recovery simulations across scenarios, sample sizes, and seeds."
    )

    parser.add_argument("--outdir", default="results/gemf_sim_recovery_energy")
    parser.add_argument("--n-list", default="50,100,200")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args()


def python_env():
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")

    if pythonpath:
        env["PYTHONPATH"] = f"{PROJECT_ROOT}:{pythonpath}"
    else:
        env["PYTHONPATH"] = str(PROJECT_ROOT)

    return env


def run(cmd, env):
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def simulation_command(data_dir, scenario, n, seed):
    return [
        sys.executable,
        "scripts/simulate_gemf_energy_data.py",
        "--outdir", str(data_dir),
        "--scenario", scenario,
        "--n", str(n),
        "--K", "20",
        "--q", "4",
        "--F", "16",
        "--C", "2",
        "--R", "3",
        "--seed", str(seed),
    ]


def fit_command(data_npz, fit_dir, ranks, args, seed):
    cmd = [
        sys.executable,
        "scripts/fit_fixed_rank_gemf.py",
        "--data", str(data_npz),
        "--outdir", str(fit_dir),
        "--R", "3",
        "--Ls", str(ranks["Ls"]),
        "--Lf", str(ranks["Lf"]),
        "--Le", str(ranks["Le"]),
        "--lambda-s", "0.01",
        "--lambda-f", "0.01",
        "--lambda-e", "0.01",
        "--zeta-e", "0.01",
        "--specific-ridge", "1e-6",
        "--score-ridge", "0.1",
        "--max-iter", str(args.max_iter),
        "--tol", "1e-5",
        "--seed", str(seed),
    ]

    if args.quiet:
        cmd.append("--quiet")

    return cmd


def evaluation_command(data_npz, fit_npz, eval_dir):
    return [
        sys.executable,
        "scripts/evaluate_gemf_simulation_fit.py",
        "--data", str(data_npz),
        "--fit", str(fit_npz),
        "--outdir", str(eval_dir),
    ]


def main():
    args = parse_args()
    env = python_env()

    out_root = PROJECT_ROOT / args.outdir
    out_root.mkdir(parents=True, exist_ok=True)

    n_list = [int(x) for x in args.n_list.split(",") if x.strip()]
    manifest = []

    for scenario, ranks in SCENARIOS.items():
        for n in n_list:
            for seed in range(args.seed_start, args.seed_end + 1):
                seed_tag = f"seed_{seed:03d}"

                run_dir = out_root / scenario / f"n{n}" / seed_tag
                data_dir = run_dir / "data"
                fit_dir = run_dir / "gemf_true_rank"
                eval_dir = run_dir / "evaluation"

                run(simulation_command(data_dir, scenario, n, seed), env)

                data_npz = data_dir / "simulated_multiview_data.npz"
                fit_npz = fit_dir / "fit_results.npz"

                run(fit_command(data_npz, fit_dir, ranks, args, seed), env)
                run(evaluation_command(data_npz, fit_npz, eval_dir), env)

                manifest.append(
                    {
                        "scenario": scenario,
                        "n": n,
                        "seed": seed,
                        "data_npz": str(data_npz),
                        "fit_npz": str(fit_npz),
                        "eval_json": str(eval_dir / "simulation_recovery_summary.json"),
                    }
                )

    manifest_file = out_root / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Wrote:", manifest_file)


if __name__ == "__main__":
    main()