#!/usr/bin/env python3
"""Create histograms from precomputed transmission statistics."""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_histogram(data):
    """Create histogram from transmission statistics."""
    # Get simulation parameters
    p_udp = data['params']['p_udp']
    n_runs = data['params']['n_runs']
    summary = data['summary']
    
    # Setup figure with LaTeX
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern'],
        'text.latex.preamble': r'\usepackage{amsmath}',
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define strategies and their styles with LaTeX names
    strategies = {
        'graph': {'name': r'\textbf{Graph Shortest Path}', 'color': '#2E8B57'},
        'mdp': {'name': r'\textbf{MDP Policy}', 'color': '#FF6B35'},
        'estimated': {'name': r'\textbf{Estimated Heuristic}', 'color': '#4169E1'},
        'expected_receivers': {'name': r'\textbf{Expected Receivers}', 'color': '#9932CC'}
    }
    
    # Calculate bar layout
    bar_width = 0.2
    active_strategies = [s for s in strategies.keys() if s in summary]
    n_strategies = len(active_strategies)
    
    # Find total width needed
    max_trans = 0
    for strategy in active_strategies:
        stats = summary[strategy]
        max_trans = max(max_trans, len(stats['tcp_pdf']))
    
    # Create x positions
    x = np.arange(max_trans)  # Transmission numbers 1 to max
    
    # Plot each strategy
    legend_elements = []
    for idx, strategy in enumerate(active_strategies):
        stats = summary[strategy]
        style = strategies[strategy]
        
        # Use PDFs directly (they are already probabilities)
        tcp_probs = np.array(stats['tcp_pdf'])
        udp_probs = np.array(stats['udp_pdf'])
        
        # Calculate x positions for this strategy
        offset = bar_width * (idx - (n_strategies-1)/2)
        x_pos = x + offset
        
        # Plot TCP and UDP stacked bars
        tcp_bars = ax.bar(x_pos[:len(tcp_probs)], tcp_probs, 
                         bar_width, 
                         color=style['color'],
                         edgecolor='black',
                         linewidth=0.5,
                         label=f"{style['name']} (TCP)")
        
        udp_bars = ax.bar(x_pos[:len(udp_probs)], udp_probs,
                         bar_width,
                         bottom=tcp_probs,
                         color=style['color'],
                         alpha=0.6,
                         hatch='//',
                         edgecolor='black',
                         linewidth=0.5,
                         label=f"{style['name']} (UDP)")
        
        # Add to legend with LaTeX labels
        legend_elements.extend([
            plt.Rectangle((0,0), 1, 1, 
                         facecolor=style['color'],
                         edgecolor='black',
                         label=f"{style['name']} " + r"\textbf{(Unicast)}"),
            plt.Rectangle((0,0), 1, 1,
                         facecolor=style['color'],
                         alpha=0.6,
                         hatch='//',
                         edgecolor='black',
                         label=f"{style['name']} " + r"\textbf{(Broadcast)}")
        ])
        
        # Print statistics
        print(f"\n{style['name']} Statistics:")
        print(f"  Mean transmissions: {stats['mean_trans']:.2f} ± {stats['std_trans']:.2f}")
        print(f"  TCP transmissions: {stats['mean_tcp']:.2f}")
        print(f"  UDP transmissions: {stats['mean_udp']:.2f}")
        print(f"  Success rate: {stats['success_rate']:.1%}")

    # X-axis ticks and labels
    # Show transmission numbers from 1 to 19
    x_min, x_max = 0, 8  # 0-based index for transmissions 1-19
    visible_x = x[x_min:x_max+1]
    ax.set_xticks(visible_x)
    ax.set_xticklabels([str(i+1) for i in range(x_min+1, x_max+2)])
    ax.set_xlim(x_min-0.5, x_max+0.5)

    # Y-axis: Show probabilities as float (0.00 to 1.00)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    
    # Labels with LaTeX
    ax.set_xlabel(r'\textbf{Total Transmission Time Required}', fontsize=16)
    ax.set_ylabel(r'\textbf{Probability}', fontsize=16)

    
    # Legend inside plot
    ax.legend(handles=legend_elements,
             loc='upper right',
             bbox_transform=ax.transAxes,
             borderaxespad=0.5,
             framealpha=0.9)
    
    # Grid and spines
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

def main():
    # Load cache
    cache_file = Path(__file__).parent / 'custom_results' / 'precomputed_stats_m1_v10.json'
    with open(cache_file, 'r') as f:
        cache = json.load(f)
    
    # Get p_udp selection
    selected = input('Enter p_udp to plot (0.2 or 0.7): ').strip() or '0.2'
    key = f"p_{float(selected):.3f}"
    
    if key not in cache:
        print(f"Error: {key} not found. Available: {list(cache.keys())}")
        return
        
    # Create and save plot
    fig = create_histogram(cache[key])
    output_path = Path(__file__).parent / 'custom_results' / f'histogram_p{cache[key]["params"]["p_udp"]:.3f}.pdf'
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"\nSaved to: {output_path}")
    plt.show()

if __name__ == '__main__':
    main()