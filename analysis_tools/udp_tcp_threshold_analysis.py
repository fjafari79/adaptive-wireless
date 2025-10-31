import numpy as np
import matplotlib.pyplot as plt

def f1(n, t):
    """First function: n/t (now omega/P_TCP)"""
    return n / t

def f2(n, p):
    """Second function: 1 + (1/p - 1)^n (now 1 + (1/P_UDP - 1)^omega)"""
    return 1 + (1/p - 1)**n

def f3(n, t, tcp_time_factor=1.2):
    """Third function: TCP time factor * n/t (now 1.2 * omega/P_TCP)"""
    return tcp_time_factor * n / t

def find_equality_curve(n, t_range):
    """Find p values where f1(omega,P_TCP) = f2(omega,P_UDP) for given P_TCP values"""
    t_values = []
    p_values = []
    
    for t in t_range:
        f1_val = f1(n, t)
        
        # Solve for p: f1_val = 1 + (1/p - 1)^n
        # (1/P_UDP - 1)^omega = f1_val - 1
        # 1/P_UDP - 1 = (f1_val - 1)^(1/omega)
        # 1/P_UDP = 1 + (f1_val - 1)^(1/omega)
        # P_UDP = 1 / (1 + (f1_val - 1)^(1/omega))
        
        if f1_val > 1:  # Only valid when f1_val > 1
            try:
                term = (f1_val - 1)**(1/n)
                p = 1 / (1 + term)
                
                # Check if p is in valid range (0, 1]
                if 0 < p <= 1:
                    t_values.append(t)
                    p_values.append(p)
            except:
                continue
    
    return np.array(t_values), np.array(p_values)

def find_equality_curve_f3(n, t_range, tcp_time_factor=1.2):
    """Find p values where f3(omega,P_TCP) = f2(omega,P_UDP) for given P_TCP values"""
    t_values = []
    p_values = []
    
    for t in t_range:
        f3_val = f3(n, t, tcp_time_factor)
        
        # Solve for p: f3_val = 1 + (1/p - 1)^n
        # Same math as find_equality_curve but using f3_val instead of f1_val
        
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
    print("Plotting equality curves where f1(\\upomega,P_TCP) = f2(\\upomega,P_UDP) and f3(\\upomega,P_TCP) = f2(\\upomega,P_UDP)")
    
    # Setup LaTeX rendering with Greek letters support
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern'],
        'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amsfonts}\usepackage{upgreek}',
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10
    })
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Define parameter ranges
    t_range = np.linspace(0.01, 1.0, 1000)
    n_values = [1, 2, 3, 4, 5, 10, 100, 1000]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'black']
    
    # FIRST PLOT: Original f1 vs f2
    ax1.set_title(r'Equality curves where $\frac{\omega}{p_{u}}= 1+\left(\frac{1}{p_{b}}-1\right)^\omega$', 
                  fontsize=14)
    
    # Collect all equality curves to find envelope for first plot
    all_t_vals_1 = []
    all_p_vals_1 = []
    curve_data_1 = []
    
    for i, n in enumerate(n_values):
        t_vals, p_vals = find_equality_curve(n, t_range)
        if len(t_vals) > 0:
            curve_data_1.append((t_vals, p_vals, n, colors[i]))
            all_t_vals_1.extend(t_vals)
            all_p_vals_1.extend(p_vals)
    
    # Create regions for first plot
    t_grid = np.linspace(0.01, 0.99, 50)
    upper_envelope_p_1 = []
    lower_envelope_p_1 = []
    
    for t in t_grid:
        p_values_at_t = []
        for t_vals, p_vals, n, color in curve_data_1:
            if len(t_vals) > 0:
                if t >= t_vals.min() and t <= t_vals.max():
                    p_interp = np.interp(t, t_vals, p_vals)
                    p_values_at_t.append(p_interp)
        
        if p_values_at_t:
            upper_envelope_p_1.append(max(p_values_at_t))
            lower_envelope_p_1.append(min(p_values_at_t))
        else:
            upper_envelope_p_1.append(0.5)
            lower_envelope_p_1.append(0.5)
    
    upper_envelope_p_1 = np.array(upper_envelope_p_1)
    lower_envelope_p_1 = np.array(lower_envelope_p_1)
    
    # Fill regions for first plot
    t_upper_1 = np.concatenate([t_grid, [1.0, 0.01]])
    p_upper_1 = np.concatenate([upper_envelope_p_1, [1.0, 1.0]])
    ax1.fill(t_upper_1, p_upper_1, color='lightblue', alpha=0.3, hatch='///', 
             edgecolor='blue', linewidth=0.5, label=r'$f_2 > f_1$ for all $\upomega$')
    
    t_lower_1 = np.concatenate([[0.01, 1.0], t_grid[::-1]])
    p_lower_1 = np.concatenate([[0.0, 0.0], lower_envelope_p_1[::-1]])
    ax1.fill(t_lower_1, p_lower_1, color='lightcoral', alpha=0.3, hatch='...', 
             edgecolor='red', linewidth=0.5, label=r'$f_1 > f_2$ for all $\upomega$')
    
    # Plot equality curves for first plot
    for t_vals, p_vals, n, color in curve_data_1:
        ax1.plot(t_vals, p_vals, color=color, linewidth=1.5, 
                label=r'$\omega=${}'.format(n), alpha=0.8)
        
        if len(t_vals) > 10:
            left_idx = len(t_vals) // 6
            label_t = t_vals[left_idx]
            label_p = p_vals[left_idx]
            
            ax1.text(label_t, label_p, r'$\omega={}$'.format(n), 
                    fontsize=8, color=color, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                             edgecolor=color, alpha=0.8, linewidth=0.5),
                    ha='center', va='center')
    
    ax1.set_xlabel(r'$p_{u}$', fontsize=14)
    ax1.set_ylabel(r'$p_{b}$', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal', adjustable='box')
    
    # SECOND PLOT: f3 vs f2 (with TCP time factor)
    ax2.set_title(r'With TCP Time Factor: $f_3(\omega,P_{TCP}) = f_2(\omega,P_{UDP})$' + '\n' + 
                  r'where $f_3 = 1.2 \times \frac{\omega}{P_{TCP}}$ and $f_2 = 1+\left(\frac{1}{P_{UDP}}-1\right)^\omega$', 
                  fontsize=14)
    
    # Collect all equality curves to find envelope for second plot
    all_t_vals_2 = []
    all_p_vals_2 = []
    curve_data_2 = []
    
    for i, n in enumerate(n_values):
        t_vals, p_vals = find_equality_curve_f3(n, t_range)
        if len(t_vals) > 0:
            curve_data_2.append((t_vals, p_vals, n, colors[i]))
            all_t_vals_2.extend(t_vals)
            all_p_vals_2.extend(p_vals)
    
    # Create regions for second plot
    upper_envelope_p_2 = []
    lower_envelope_p_2 = []
    
    for t in t_grid:
        p_values_at_t = []
        for t_vals, p_vals, n, color in curve_data_2:
            if len(t_vals) > 0:
                if t >= t_vals.min() and t <= t_vals.max():
                    p_interp = np.interp(t, t_vals, p_vals)
                    p_values_at_t.append(p_interp)
        
        if p_values_at_t:
            upper_envelope_p_2.append(max(p_values_at_t))
            lower_envelope_p_2.append(min(p_values_at_t))
        else:
            upper_envelope_p_2.append(0.5)
            lower_envelope_p_2.append(0.5)
    
    upper_envelope_p_2 = np.array(upper_envelope_p_2)
    lower_envelope_p_2 = np.array(lower_envelope_p_2)
    
    # Fill regions for second plot
    t_upper_2 = np.concatenate([t_grid, [1.0, 0.01]])
    p_upper_2 = np.concatenate([upper_envelope_p_2, [1.0, 1.0]])
    ax2.fill(t_upper_2, p_upper_2, color='lightblue', alpha=0.3, hatch='///', 
                  edgecolor='blue', linewidth=0.5, label=r'$f_2 > f_3$ for all $\upomega$')
    
    t_lower_2 = np.concatenate([[0.01, 1.0], t_grid[::-1]])
    p_lower_2 = np.concatenate([[0.0, 0.0], lower_envelope_p_2[::-1]])
    ax2.fill(t_lower_2, p_lower_2, color='lightcoral', alpha=0.3, hatch='...', 
                  edgecolor='red', linewidth=0.5, label=r'$f_3 > f_2$ for all $\upomega$')
    
    # Plot equality curves for second plot
    for t_vals, p_vals, n, color in curve_data_2:
        ax2.plot(t_vals, p_vals, color=color, linewidth=1.5, 
                label=r'$\omega=${}'.format(n), alpha=0.8)
        
        if len(t_vals) > 10:
            left_idx = len(t_vals) // 6
            label_t = t_vals[left_idx]
            label_p = p_vals[left_idx]
            
            ax2.text(label_t, label_p, r'$\omega={}$'.format(n), 
                    fontsize=8, color=color, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                             edgecolor=color, alpha=0.8, linewidth=0.5),
                    ha='center', va='center')
    
    ax2.set_xlabel(r'$p_{u}$', fontsize=14)
    ax2.set_ylabel(r'$p_{b}$', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal', adjustable='box')
    
    # Add legends to both plots
    from matplotlib.patches import Patch
    
    # Legend for first plot
    legend_elements_1 = [plt.Line2D([0], [0], color=color, lw=1.5, label=r'$\upomega=${}'.format(n)) 
                        for _, _, n, color in curve_data_1]
    legend_elements_1.append(Patch(facecolor='lightcoral', alpha=0.3, hatch='...', edgecolor='red', 
                                  label=r'$f_1 > f_2$ for all $\upomega$'))
    legend_elements_1.append(Patch(facecolor='lightblue', alpha=0.3, hatch='///', edgecolor='blue', 
                                  label=r'$f_2 > f_1$ for all $\upomega$'))
    
    ax1.legend(handles=legend_elements_1, fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Legend for second plot
    legend_elements_2 = [plt.Line2D([0], [0], color=color, lw=1.5, label=r'$\upomega=${}'.format(n)) 
                        for _, _, n, color in curve_data_2]
    legend_elements_2.append(Patch(facecolor='lightcoral', alpha=0.3, hatch='...', edgecolor='red', 
                                  label=r'$f_3 > f_2$ for all $\upomega$'))
    legend_elements_2.append(Patch(facecolor='lightblue', alpha=0.3, hatch='///', edgecolor='blue', 
                                  label=r'$f_2 > f_3$ for all $\upomega$'))
    
    ax2.legend(handles=legend_elements_2, fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Create separate figures for each subplot and save them
    # Save left plot (original curves)
    fig_left = plt.figure(figsize=(8, 8))
    ax_left = fig_left.add_subplot(111)
    ax_left.set_aspect('equal', adjustable='box')
    
    # Copy content from ax1 to new figure
    for line in ax1.get_lines():
        ax_left.plot(line.get_xdata(), line.get_ydata(), 
                    color=line.get_color(), linewidth=line.get_linewidth(),
                    alpha=line.get_alpha(), label=line.get_label())
    
    # Copy filled regions for first plot
    t_upper_1 = np.concatenate([t_grid, [1.0, 0.01]])
    p_upper_1 = np.concatenate([upper_envelope_p_1, [1.0, 1.0]])
    ax_left.fill(t_upper_1, p_upper_1, color='lightblue', alpha=0.3, hatch='///', 
                  edgecolor='blue', linewidth=0.5, label=r'$f_2 > f_3$ for all $\upomega$')
    
    t_lower_1 = np.concatenate([[0.01, 1.0], t_grid[::-1]])
    p_lower_1 = np.concatenate([[0.0, 0.0], lower_envelope_p_1[::-1]])
    ax_right.fill(t_lower_2, p_lower_2, color='lightcoral', alpha=0.3, hatch='...', 
                  edgecolor='red', linewidth=0.5, label=r'$f_3 > f_2$ for all $\upomega$')
    
    # Copy texts
    for text in ax1.texts:
        bbox_props = dict(boxstyle='round,pad=0.2', 
                         facecolor='white', 
                         edgecolor=text.get_color(),
                         alpha=0.8,
                         linewidth=0.5)
        ax_left.text(text.get_position()[0], text.get_position()[1],
                    text.get_text().replace('ω', r'$\upomega$'), fontsize=text.get_fontsize(),
                    color=text.get_color(), weight=text.get_weight(),
                    bbox=bbox_props,
                    ha=text.get_ha(), va=text.get_va())
    
    # Copy axes properties
    ax_left.set_xlabel(ax1.get_xlabel())
    ax_left.set_ylabel(ax1.get_ylabel())
    ax_left.set_title(ax1.get_title())
    ax_left.grid(True, alpha=0.3)
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(0, 1)
    ax_left.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save left plot
    output_path_left = 'threshold_analysis_curves_original.pdf'
    plt.figure(fig_left.number)
    plt.savefig(output_path_left, bbox_inches='tight', dpi=300)
    print(f"\nSaved original curves to: {output_path_left}")
    
    # Save right plot (modified curves with TCP time factor)
    fig_right = plt.figure(figsize=(8, 8))
    ax_right = fig_right.add_subplot(111)
    ax_right.set_aspect('equal', adjustable='box')
    
    # Copy content from ax2 to new figure
    for line in ax2.get_lines():
        ax_right.plot(line.get_xdata(), line.get_ydata(), 
                     color=line.get_color(), linewidth=line.get_linewidth(),
                     alpha=line.get_alpha(), label=line.get_label())
    
    # Copy filled regions for second plot
    t_upper_2 = np.concatenate([t_grid, [1.0, 0.01]])
    p_upper_2 = np.concatenate([upper_envelope_p_2, [1.0, 1.0]])
    ax_right.fill(t_upper_2, p_upper_2, color='lightblue', alpha=0.3, hatch='///', 
                  edgecolor='blue', linewidth=0.5, label=r'$f_2 > f_3$ for all $\upomega$')
    
    t_lower_2 = np.concatenate([[0.01, 1.0], t_grid[::-1]])
    p_lower_2 = np.concatenate([[0.0, 0.0], lower_envelope_p_2[::-1]])
    ax_right.fill(t_lower_2, p_lower_2, color='lightcoral', alpha=0.3, hatch='...', 
                  edgecolor='red', linewidth=0.5, label=r'$f_3 > f_2$ for all $\upomega$')
    
    # Copy texts
    for text in ax2.texts:
        bbox_props = dict(boxstyle='round,pad=0.2', 
                         facecolor='white', 
                         edgecolor=text.get_color(),
                         alpha=0.8,
                         linewidth=0.5)
        ax_right.text(text.get_position()[0], text.get_position()[1],
                     text.get_text().replace('ω', r'$\upomega$'), fontsize=text.get_fontsize(),
                     color=text.get_color(), weight=text.get_weight(),
                     bbox=bbox_props,
                     ha=text.get_ha(), va=text.get_va())
    
    # Copy axes properties
    ax_right.set_xlabel(ax2.get_xlabel())
    ax_right.set_ylabel(ax2.get_ylabel())
    ax_right.set_title(ax2.get_title())
    ax_right.grid(True, alpha=0.3)
    ax_right.set_xlim(0, 1)
    ax_right.set_ylim(0, 1)
    ax_right.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save right plot
    output_path_right = 'threshold_analysis_curves_with_factor.pdf'
    plt.figure(fig_right.number)
    plt.savefig(output_path_right, bbox_inches='tight', dpi=300)
    print(f"Saved modified curves to: {output_path_right}")
    
    # Show combined plot
    plt.figure(fig.number)
    plt.show()
    
    print("\nPlot displayed showing:")
    print("Left: Original curves where f1 = f2 for different ω values")
    print("Right: Modified curves where f3 = f2 (with TCP time factor 1.2) for different ω values")

if __name__ == "__main__":
    main()