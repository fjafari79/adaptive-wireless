#!/usr/bin/env python3
"""Precompute statistics for chosen parameter sets and save to JSON.

This script runs the simulation for two p_udp values (0.2 and 0.7) with
m=1, v=10, p_tcp=0.9 and num_runs=10000 and stores aggregated statistics
and PDF arrays into a JSON file under `custom_results` so plots can be
generated later without rerunning the full simulation.
"""
import json
import os
import sys
from pathlib import Path
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# Ensure we can import the modified_simulation script modules
sys.path.append(str(Path(__file__).parent))

from custom_parameter_comparison import run_custom_simulation


def summarize_results(results):
    # Build a compact summary per strategy
    summary = {}
    for strategy, data in results.items():
        total_times = data['total_times']
        transmissions = data['transmissions']
        tcp = data['tcp_transmissions']
        udp = data['udp_transmissions']
        n = len(transmissions)
        if n == 0:
            continue
            
        import numpy as np
        
        # Basic statistics
        mean_trans = float(np.mean(transmissions))
        std_trans = float(np.std(transmissions))
        mean_tcp = float(np.mean(tcp))
        mean_udp = float(np.mean(udp))
        success_rate = float(data['success_count']) / n

        # Find maximum number of transmissions
        max_trans = max(transmissions) if transmissions else 1
        
        # Initialize PDF arrays with zeros (one bin per transmission count)
        n_bins = max_trans + 1  # +1 because we include 0 transmissions
        total_pdf = [0.0] * n_bins
        tcp_pdf = [0.0] * n_bins
        udp_pdf = [0.0] * n_bins
        
        # Build PDFs based on actual transmission counts
        for i in range(len(transmissions)):
            total_count = transmissions[i]
            tcp_count = tcp[i]
            udp_count = udp[i]
            
            # Add to the appropriate bin (total_count is the bin index)
            total_pdf[total_count] += 1
            
            # Split between TCP and UDP based on actual counts
            if total_count > 0:
                tcp_pdf[total_count] += tcp_count / total_count
                udp_pdf[total_count] += udp_count / total_count

        # Normalize PDFs
        total_pdf = [v / n for v in total_pdf]
        tcp_pdf = [v / n for v in tcp_pdf]
        udp_pdf = [v / n for v in udp_pdf]

        summary[strategy] = {
            'n_runs': n,
            'mean_trans': mean_trans,
            'std_trans': std_trans,
            'mean_tcp': mean_tcp,
            'mean_udp': mean_udp,
            'success_rate': success_rate,
            'total_pdf': total_pdf,
            'tcp_pdf': tcp_pdf,
            'udp_pdf': udp_pdf,
            'mean_time': float(np.mean(total_times)),
            'std_time': float(np.std(total_times))
        }
    return summary


def main():
    n_messages = 1
    n_vehicles = 10
    p_tcp = 0.9
    p_udps = [0.2, 0.7]
    num_runs = 10000
    initial_state_type = "all_ones"  # Start with all vehicles needing messages
    initial_state_params = {}

    results_dir = Path(__file__).parent / 'custom_results'
    results_dir.mkdir(parents=True, exist_ok=True)

    cache = {}
    for p_udp in p_udps:
        print(f"Running simulations for p_udp={p_udp} ...")
        # run_custom_simulation is in custom_parameter_comparison.py
        results = run_custom_simulation(n_messages, n_vehicles, p_tcp, p_udp, num_runs,
                                     initial_state_type=initial_state_type, 
                                     initial_state_params=initial_state_params)
        summary = summarize_results(results)
        cache[f"p_{p_udp:.3f}"] = {
            'params': {'n_messages': n_messages, 'n_vehicles': n_vehicles, 'p_tcp': p_tcp, 'p_udp': p_udp, 'n_runs': num_runs},
            'summary': summary
        }

    out_file = results_dir / 'precomputed_stats_m1_v10.json'
    with open(out_file, 'w') as f:
        json.dump(cache, f, indent=2)

    print(f"Saved precomputed stats to: {out_file}")


if __name__ == '__main__':
    main()
