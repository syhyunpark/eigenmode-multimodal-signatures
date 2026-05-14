#!/usr/bin/env python3
# script: run_gemf_rank_selection_simulations.py

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
        description="Run GEMF rank-selection simulations across scenarios, sample sizes, and seeds."
    )

    parser.add_argument("--data_root", default="results/gemf_sim_recovery_energy_final")
    parser.add_argument("--outdir", default="results/gemf_sim_rank_selection_energy")

    parser.add_argument("--n-list", default="100")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=100)

    parser.add_argument("--R_grid", default="1,2,3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--max_iter", type=int, default=50)
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


def data_file_for(data_root, scenario, n, seed):
    return (
        PROJECT_ROOT
        / data_root
        / scenario
        / f"n{n}"
        / f"seed_{seed:03d}"
        / "data"
        / "simulated_multiview_data.npz"
    )


def make_command(data_npz, outdir, ranks, args, seed):
    cmd = [
        sys.executable,
        "scripts/select_gemf_rank_by_reconstruction.py",
        "--data", str(data_npz),
        "--outdir", str(outdir),
        "--R_grid", args.R_grid,
        "--Ls", str(ranks["Ls"]),
        "--Lf", str(ranks["Lf"]),
        "--Le", str(ranks["Le"]),
        "--n_folds", str(args.n_folds),
        "--seed", str(seed),
        "--score_ridge", "0.1",
        "--max_iter", str(args.max_iter),
    ]

    if args.quiet:
        cmd.append("--quiet")

    return cmd


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

                data_npz = data_file_for(args.data_root, scenario, n, seed)
                if not data_npz.exists():
                    raise FileNotFoundError(
                        f"Missing data file: {data_npz}\n"
                        "Run the recovery simulations first, or check --data_root."
                    )

                run_outdir = out_root / scenario / f"n{n}" / seed_tag

                cmd = make_command(data_npz, run_outdir, ranks, args, seed)
                run(cmd, env)

                manifest.append(
                    {
                        "scenario": scenario,
                        "n": n,
                        "seed": seed,
                        "true_R": 3,
                        "selection_json": str(run_outdir / "selected_rank_summary.json"),
                        "cv_rank_summary": str(run_outdir / "cv_rank_summary.csv"),
                        "cv_fold_results": str(run_outdir / "cv_fold_results.csv"),
                    }
                )

    manifest_file = out_root / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Wrote:", manifest_file)


if __name__ == "__main__":
    main()