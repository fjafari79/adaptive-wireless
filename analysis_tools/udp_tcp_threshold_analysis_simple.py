import numpy as np
import matplotlib.pyplot as plt

def f2(n, p):
    """Second function: 1 + (1/p - 1)^n (now 1 + (1/P_UDP - 1)^omega)"""
    return 1 + (1/p - 1)**n

def f3(n, t, tcp_time_factor=1.2):
    """Third function: TCP time factor * n/t (now 1.2 * omega/P_TCP)"""
    return tcp_time_factor * n / t

def find_equality_curve_f3(n, t_range, tcp_time_factor=1.2):
    """Find p values where f3(omega,P_TCP) = f2(omega,P_UDP) for given P_TCP values"""
    t_values = []
    p_values = []
    
    for t in t_range:
        f3_val = f3(n, t, tcp_time_factor)
        
        if f3_val > 1:  # Only valid when f3_val > 1
            try:
                term = (f3_val - 1)**(1/n)
                p = 1 / (1 + term)
                
                # Check if p is in valid range (0, 1]
                if 0 < p <= 1:
                    t_values.append(t)
                    p_values.append(p)
            except:
                continue
    
    return np.array(t_values), np.array(p_values)

def main():
    print("Plotting equality curves where f3(omega,P_TCP) = f2(omega,P_UDP)")
    
    # Setup LaTeX rendering with Greek letters support
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern'],
        'text.latex.preamble': r'\usepackage{amsmath}\usepackage{textcomp}\usepackage[utf8]{inputenc}\usepackage{upgreek}',
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10
    })
    
    # Create figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    
    # Define parameter ranges
    t_range = np.linspace(0.01, 1.0, 1000)
    n_values = [1, 2, 3, 4, 5, 10, 100, 1000]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'black']
    
    # Set plot title
    ax.set_title(r'With TCP Time Factor: $f_3(\omega,P_{TCP}) = f_2(\omega,P_{UDP})$' + '\n' + 
                r'where $f_3 = 1.2 \times \frac{\omega}{P_{TCP}}$ and $f_2 = 1+\left(\frac{1}{P_{UDP}}-1\right)^\omega$', 
                fontsize=14)
    
    # Collect all equality curves to find envelope
    curve_data = []
    t_grid = np.linspace(0.01, 0.99, 50)
    upper_envelope_p = []
    lower_envelope_p = []
    
    for i, n in enumerate(n_values):
        t_vals, p_vals = find_equality_curve_f3(n, t_range)
        if len(t_vals) > 0:
            curve_data.append((t_vals, p_vals, n, colors[i]))
    
    # Create regions
    for t in t_grid:
        p_values_at_t = []
        for t_vals, p_vals, n, color in curve_data:
            if len(t_vals) > 0:
                if t >= t_vals.min() and t <= t_vals.max():
                    p_interp = np.interp(t, t_vals, p_vals)
                    p_values_at_t.append(p_interp)
        
        if p_values_at_t:
            upper_envelope_p.append(max(p_values_at_t))
            lower_envelope_p.append(min(p_values_at_t))
        else:
            upper_envelope_p.append(0.5)
            lower_envelope_p.append(0.5)
    
    upper_envelope_p = np.array(upper_envelope_p)
    lower_envelope_p = np.array(lower_envelope_p)
    
    # Fill regions
    t_upper = np.concatenate([t_grid, [1.0, 0.01]])
    p_upper = np.concatenate([upper_envelope_p, [1.0, 1.0]])
    ax.fill(t_upper, p_upper, color='lightblue', alpha=0.3, hatch='///', 
            edgecolor='blue', linewidth=0.5, label=r'$f_2 > f_3$ for all $\omega$')
    
    t_lower = np.concatenate([[0.01, 1.0], t_grid[::-1]])
    p_lower = np.concatenate([[0.0, 0.0], lower_envelope_p[::-1]])
    ax.fill(t_lower, p_lower, color='lightcoral', alpha=0.3, hatch='...', 
            edgecolor='red', linewidth=0.5, label=r'$f_3 > f_2$ for all $\omega$')
    
    # Plot equality curves
    for t_vals, p_vals, n, color in curve_data:
        ax.plot(t_vals, p_vals, color=color, linewidth=1.5, 
               label=r'$\omega=' + str(n) + '$', alpha=0.8)
        
        if len(t_vals) > 10:
            left_idx = len(t_vals) // 6
            label_t = t_vals[left_idx]
            label_p = p_vals[left_idx]
            
            ax.text(label_t, label_p, r'$\omega=' + str(n) + '$', 
                   fontsize=8, color=color, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                            edgecolor=color, alpha=0.8, linewidth=0.5),
                   ha='center', va='center')
    
    # Set labels and grid
    ax.set_xlabel(r'$p_{u}$', fontsize=14)
    ax.set_ylabel(r'$p_{b}$', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Add legend
    ax.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save plot
    output_path = 'threshold_analysis_curves_with_factor.pdf'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"\nSaved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main()