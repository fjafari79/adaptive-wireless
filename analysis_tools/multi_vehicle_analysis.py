import numpy as np
import matplotlib.pyplot as plt
from weighted_graph import *
import time
from itertools import product

class MultiVehicleThresholdAnalyzer:
    """
    Analyze TCP vs UDP preferences with increasing numbers of vehicles.
    Start with 1 message and systematically increase vehicle count.
    """
    
    def __init__(self, max_vehicles=6):
        """
        Initialize the multi-vehicle analyzer.
        
        Args:
            max_vehicles (int): Maximum number of vehicles to analyze
        """
        self.max_vehicles = max_vehicles
        self.results_by_config = {}
        
    def analyze_configuration(self, n_messages, n_vehicles, p_tcp_range=(0.1, 0.99), 
                            p_udp_range=(0.1, 0.99), resolution=0.05):
        """
        Analyze a specific (messages, vehicles) configuration.
        
        Args:
            n_messages (int): Number of messages
            n_vehicles (int): Number of vehicles
            p_tcp_range (tuple): TCP probability range
            p_udp_range (tuple): UDP probability range
            resolution (float): Analysis resolution
            
        Returns:
            dict: Analysis results for this configuration
        """
        print(f"=== ANALYZING CONFIGURATION: {n_messages} MESSAGE(S), {n_vehicles} VEHICLE(S) ===")
        print(f"TCP range: {p_tcp_range}, UDP range: {p_udp_range}, Resolution: {resolution}")
        
        start_time = time.time()
        
        # Generate probability grids
        p_tcp_values = np.arange(p_tcp_range[0], p_tcp_range[1] + resolution, resolution)
        p_udp_values = np.arange(p_udp_range[0], p_udp_range[1] + resolution, resolution)
        
        # Results storage
        tcp_preferred_points = []
        udp_preferred_points = []
        equal_points = []
        error_points = []
        
        total_combinations = len(p_tcp_values) * len(p_udp_values)
        analyzed = 0
        
        print(f"Analyzing {total_combinations} probability combinations...")
        
        for p_tcp in p_tcp_values:
            for p_udp in p_udp_values:
                preference = self._analyze_single_combination(n_messages, n_vehicles, p_tcp, p_udp)
                
                if preference == 'TCP':
                    tcp_preferred_points.append((p_tcp, p_udp))
                elif preference == 'UDP':
                    udp_preferred_points.append((p_tcp, p_udp))
                elif preference == 'EQUAL':
                    equal_points.append((p_tcp, p_udp))
                else:  # ERROR
                    error_points.append((p_tcp, p_udp))
                
                analyzed += 1
                if analyzed % 500 == 0:
                    progress = 100 * analyzed / total_combinations
                    elapsed = time.time() - start_time
                    eta = elapsed * (total_combinations - analyzed) / analyzed if analyzed > 0 else 0
                    print(f"  Progress: {analyzed}/{total_combinations} ({progress:.1f}%) "
                          f"- Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
        
        elapsed_time = time.time() - start_time
        
        # Calculate switching boundaries
        boundaries = self._find_switching_boundaries(tcp_preferred_points, udp_preferred_points, resolution)
        
        # Calculate statistics
        config_key = f"{n_messages}m_{n_vehicles}v"
        results = {
            'config': config_key,
            'n_messages': n_messages,
            'n_vehicles': n_vehicles,
            'p_tcp_range': p_tcp_range,
            'p_udp_range': p_udp_range,
            'resolution': resolution,
            'tcp_preferred_points': tcp_preferred_points,
            'udp_preferred_points': udp_preferred_points,
            'equal_points': equal_points,
            'error_points': error_points,
            'switching_boundaries': boundaries,
            'total_combinations': total_combinations,
            'analysis_time': elapsed_time,
            'tcp_preference_ratio': len(tcp_preferred_points) / total_combinations,
            'udp_preference_ratio': len(udp_preferred_points) / total_combinations,
            'equal_ratio': len(equal_points) / total_combinations,
            'error_ratio': len(error_points) / total_combinations
        }
        
        # Store results
        self.results_by_config[config_key] = results
        
        print(f"Configuration analysis completed in {elapsed_time:.2f}s")
        print(f"TCP preferred: {len(tcp_preferred_points)} ({100*results['tcp_preference_ratio']:.1f}%)")
        print(f"UDP preferred: {len(udp_preferred_points)} ({100*results['udp_preference_ratio']:.1f}%)")
        print(f"Equal: {len(equal_points)} ({100*results['equal_ratio']:.1f}%)")
        print(f"Errors: {len(error_points)} ({100*results['error_ratio']:.1f}%)")
        print(f"Boundary points: {len(boundaries)}")
        print()
        
        return results
    
    def _analyze_single_combination(self, n_messages, n_vehicles, p_tcp, p_udp):
        """
        Analyze a single probability combination for given configuration.
        
        Args:
            n_messages (int): Number of messages
            n_vehicles (int): Number of vehicles
            p_tcp (float): TCP success probability
            p_udp (float): UDP success probability
            
        Returns:
            str: 'TCP', 'UDP', 'EQUAL', or 'ERROR'
        """
        try:
            # For small configurations, we can use direct calculation
            if n_messages == 1:
                return self._analyze_single_message_case(n_vehicles, p_tcp, p_udp)
            else:
                # For multi-message cases, we need to use the graph approach
                return self._analyze_multi_message_case(n_messages, n_vehicles, p_tcp, p_udp)
                
        except Exception as e:
            return 'ERROR'
    
    def _analyze_single_message_case(self, n_vehicles, p_tcp, p_udp):
        """
        Analyze single message case with multiple vehicles using direct calculation.
        
        For 1 message to n vehicles, the transition is from [1,1,...,1] to some subset.
        The optimal strategy depends on the number of vehicles that need to be reached.
        """
        try:
            # For single message case, we analyze the most common transition:
            # from all vehicles having the message to no vehicles having it
            # This represents the maximum delta (all vehicles flip from 1 to 0)
            
            delta = n_vehicles  # All vehicles flip from 1 to 0
            
            # Calculate weights
            if p_udp == 0:
                udp_weight = float('inf')
            else:
                udp_weight = 1 + (1/p_udp - 1)**delta
            
            if p_tcp == 0:
                tcp_weight = float('inf')
            else:
                tcp_weight = 1 / p_tcp
            
            # TCP is only applicable for single-flip transitions (delta=1)
            # For multi-vehicle cases where delta > 1, only UDP applies
            if delta == 1:
                # Single vehicle case - both TCP and UDP apply
                if tcp_weight < udp_weight:
                    return 'TCP'
                elif udp_weight < tcp_weight:
                    return 'UDP'
                else:
                    return 'EQUAL'
            else:
                # Multi-vehicle case - only UDP applies, TCP not available
                # We compare UDP weight vs hypothetical TCP weight to understand preference
                if tcp_weight < udp_weight:
                    return 'TCP'  # TCP would be preferred if it were available
                else:
                    return 'UDP'
                    
        except Exception as e:
            return 'ERROR'
    
    def _analyze_multi_message_case(self, n_messages, n_vehicles, p_tcp, p_udp):
        """
        Analyze multi-message case using graph-based approach.
        
        This is more complex and requires building the actual state graph.
        """
        try:
            # Set up the environment
            import weighted_graph
            weighted_graph.n_messages = n_messages
            weighted_graph.n_vehicles = n_vehicles
            weighted_graph.matrix_shape = (n_messages, n_vehicles)
            weighted_graph.p_tcp = p_tcp
            weighted_graph.p_udp = p_udp
            
            # Create initial state matrix (all messages to all vehicles)
            initial_matrix = np.ones((n_messages, n_vehicles), dtype=int)
            
            # Initialize states dictionary
            states = generate_states()
            weighted_graph.states_dict = {matrix_to_index(m): (m, total) for m, total in states}
            
            # Create restricted graph
            restricted_graph, restricted_states = create_restricted_graph(initial_matrix)
            
            # Update global states
            original_states_dict = weighted_graph.states_dict.copy()
            weighted_graph.states_dict = restricted_states
            
            # Assign weights
            weighted_graph_obj = assign_edge_weights(restricted_graph)
            
            # Find initial state
            initial_state_id = None
            for state_id, (matrix, _) in restricted_states.items():
                if np.array_equal(matrix, initial_matrix):
                    initial_state_id = state_id
                    break
            
            if initial_state_id is None:
                weighted_graph.states_dict = original_states_dict
                return 'ERROR'
            
            # Analyze available actions
            tcp_weights = []
            udp_weights = []
            
            for neighbor in weighted_graph_obj.neighbors(initial_state_id):
                edge_data = weighted_graph_obj[initial_state_id][neighbor]
                action_type = edge_data.get('action_type', 1)
                weight = edge_data.get('weight', float('inf'))
                
                if action_type == 0:  # TCP
                    tcp_weights.append(weight)
                elif action_type == 1:  # UDP
                    udp_weights.append(weight)
            
            # Restore states
            weighted_graph.states_dict = original_states_dict
            
            # Determine preference based on available actions
            min_tcp = min(tcp_weights) if tcp_weights else float('inf')
            min_udp = min(udp_weights) if udp_weights else float('inf')
            
            if min_tcp < min_udp:
                return 'TCP'
            elif min_udp < min_tcp:
                return 'UDP'
            else:
                return 'EQUAL'
                
        except Exception as e:
            return 'ERROR'
    
    def _find_switching_boundaries(self, tcp_points, udp_points, resolution):
        """Find boundary points where preferences switch."""
        boundaries = []
        tcp_set = set(tcp_points)
        udp_set = set(udp_points)
        
        for p_tcp, p_udp in tcp_points:
            neighbors = [
                (p_tcp + resolution, p_udp),
                (p_tcp - resolution, p_udp),
                (p_tcp, p_udp + resolution),
                (p_tcp, p_udp - resolution)
            ]
            
            for neighbor in neighbors:
                if neighbor in udp_set:
                    boundaries.append(((p_tcp, p_udp), neighbor))
        
        return boundaries
    
    def run_progressive_analysis(self, n_messages=1, max_vehicles=None, resolution=0.05):
        """
        Run progressive analysis with increasing vehicle counts.
        
        Args:
            n_messages (int): Number of messages (start with 1)
            max_vehicles (int): Maximum number of vehicles (uses self.max_vehicles if None)
            resolution (float): Analysis resolution
            
        Returns:
            dict: Combined results for all configurations
        """
        if max_vehicles is None:
            max_vehicles = self.max_vehicles
            
        print(f"=== PROGRESSIVE VEHICLE ANALYSIS ===")
        print(f"Configuration: {n_messages} message(s), 1 to {max_vehicles} vehicles")
        print(f"Resolution: {resolution}")
        print()
        
        all_results = {}
        
        for n_vehicles in range(1, max_vehicles + 1):
            print(f"\n--- VEHICLE COUNT: {n_vehicles} ---")
            
            # Adjust resolution based on complexity
            # Use finer resolution for smaller configurations, coarser for larger ones
            if n_vehicles <= 2:
                adjusted_resolution = resolution  # Full resolution for simple cases
            elif n_vehicles <= 4:
                adjusted_resolution = resolution * 1.5  # Slightly coarser
            else:
                adjusted_resolution = resolution * 2.0  # Coarser for complex cases
            
            # Limit analysis range for very complex cases to manage computation time
            if n_vehicles > 4:
                p_tcp_range = (0.2, 0.9)
                p_udp_range = (0.2, 0.9)
                print(f"  Using reduced probability range for {n_vehicles} vehicles: {p_tcp_range}")
            else:
                p_tcp_range = (0.1, 0.99)
                p_udp_range = (0.1, 0.99)
            
            results = self.analyze_configuration(
                n_messages=n_messages,
                n_vehicles=n_vehicles,
                p_tcp_range=p_tcp_range,
                p_udp_range=p_udp_range,
                resolution=adjusted_resolution
            )
            
            config_key = f"{n_messages}m_{n_vehicles}v"
            all_results[config_key] = results
        
        return all_results
    
    def plot_configuration_comparison(self, configs_to_plot=None, save_plot=True):
        """
        Create comparison plots for different configurations.
        
        Args:
            configs_to_plot (list): List of configuration keys to plot
            save_plot (bool): Whether to save plots
        """
        if not self.results_by_config:
            print("No results to plot. Run analysis first.")
            return
        
        if configs_to_plot is None:
            configs_to_plot = list(self.results_by_config.keys())
        
        # Create subplot grid
        n_configs = len(configs_to_plot)
        cols = min(3, n_configs)
        rows = (n_configs + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_configs == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, config_key in enumerate(configs_to_plot):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            results = self.results_by_config[config_key]
            
            # Extract data
            tcp_points = np.array(results['tcp_preferred_points'])
            udp_points = np.array(results['udp_preferred_points'])
            
            # Plot TCP preferred region
            if len(tcp_points) > 0:
                ax.scatter(tcp_points[:, 0], tcp_points[:, 1], 
                          c='blue', alpha=0.6, s=10, label='TCP Preferred', marker='s')
            
            # Plot UDP preferred region
            if len(udp_points) > 0:
                ax.scatter(udp_points[:, 0], udp_points[:, 1], 
                          c='orange', alpha=0.6, s=10, label='UDP Preferred', marker='o')
            
            # Add diagonal reference line
            ax.plot([0.1, 0.99], [0.1, 0.99], 'k--', alpha=0.5, label='p_tcp = p_udp')
            
            ax.set_xlabel('TCP Probability')
            ax.set_ylabel('UDP Probability')
            ax.set_title(f'{results["n_messages"]}M, {results["n_vehicles"]}V\n'
                        f'TCP: {100*results["tcp_preference_ratio"]:.1f}%, '
                        f'UDP: {100*results["udp_preference_ratio"]:.1f}%')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(results['p_tcp_range'])
            ax.set_ylim(results['p_udp_range'])
        
        # Hide unused subplots
        for idx in range(n_configs, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_plot:
            filename = f'multi_vehicle_comparison_{n_configs}configs.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved as {filename}")
        
        plt.show()
    
    def generate_summary_report(self):
        """Generate a summary report of all analyzed configurations."""
        if not self.results_by_config:
            print("No results to summarize. Run analysis first.")
            return
        
        print("=== MULTI-VEHICLE THRESHOLD ANALYSIS SUMMARY ===")
        print()
        
        # Sort configurations by number of vehicles
        sorted_configs = sorted(self.results_by_config.items(), 
                               key=lambda x: (x[1]['n_messages'], x[1]['n_vehicles']))
        
        print("Configuration | TCP% | UDP% | Equal% | Error% | Boundaries | Time(s)")
        print("-" * 70)
        
        for config_key, results in sorted_configs:
            tcp_pct = 100 * results['tcp_preference_ratio']
            udp_pct = 100 * results['udp_preference_ratio']
            equal_pct = 100 * results['equal_ratio']
            error_pct = 100 * results['error_ratio']
            n_boundaries = len(results['switching_boundaries'])
            analysis_time = results['analysis_time']
            
            print(f"{config_key:12} | {tcp_pct:4.1f} | {udp_pct:4.1f} | {equal_pct:5.1f} | "
                  f"{error_pct:5.1f} | {n_boundaries:9} | {analysis_time:6.1f}")
        
        print()
        
        # Analysis insights
        print("=== KEY INSIGHTS ===")
        
        # Find trend in TCP preference as vehicles increase
        single_msg_configs = [(k, v) for k, v in sorted_configs if v['n_messages'] == 1]
        
        if len(single_msg_configs) > 1:
            print("TCP Preference Trend (1 message, increasing vehicles):")
            for config_key, results in single_msg_configs:
                n_vehicles = results['n_vehicles']
                tcp_pct = 100 * results['tcp_preference_ratio']
                print(f"  {n_vehicles} vehicle(s): {tcp_pct:.1f}% TCP preference")
        
        print()

    def analyze_switching_threshold_trend(self, n_messages=1, max_vehicles=None, fixed_param='p_tcp', fixed_value=0.8):
        """
        Analyze how the switching threshold changes with increasing vehicle count.
        
        Args:
            n_messages (int): Number of messages
            max_vehicles (int): Maximum vehicles to analyze
            fixed_param (str): 'p_tcp' or 'p_udp' - parameter to fix
            fixed_value (float): Value of the fixed parameter
            
        Returns:
            dict: Switching threshold data for each vehicle count
        """
        if max_vehicles is None:
            max_vehicles = self.max_vehicles
            
        print(f"=== SWITCHING THRESHOLD TREND ANALYSIS ===")
        print(f"Configuration: {n_messages} message(s), 1 to {max_vehicles} vehicles")
        print(f"Fixed: {fixed_param} = {fixed_value}")
        print()
        
        threshold_data = {}
        
        for n_vehicles in range(1, max_vehicles + 1):
            print(f"Finding switching threshold for {n_vehicles} vehicle(s)...")
            
            # Binary search for switching point
            if fixed_param == 'p_tcp':
                threshold = self._find_switching_point_binary_search(
                    n_messages, n_vehicles, p_tcp_fixed=fixed_value
                )
            else:
                threshold = self._find_switching_point_binary_search(
                    n_messages, n_vehicles, p_udp_fixed=fixed_value
                )
            
            threshold_data[n_vehicles] = {
                'switching_point': threshold,
                'fixed_param': fixed_param,
                'fixed_value': fixed_value
            }
            
            if threshold is not None:
                print(f"  Switching point: {threshold:.4f}")
            else:
                print(f"  No switching point found in range")
        
        return threshold_data
    
    def _find_switching_point_binary_search(self, n_messages, n_vehicles, 
                                          p_tcp_fixed=None, p_udp_fixed=None, 
                                          search_range=(0.1, 0.99), tolerance=1e-4):
        """
        Use binary search to find the exact switching point for a configuration.
        """
        def get_preference(varying_value):
            if p_tcp_fixed is not None:
                return self._analyze_single_combination(n_messages, n_vehicles, p_tcp_fixed, varying_value)
            else:
                return self._analyze_single_combination(n_messages, n_vehicles, varying_value, p_udp_fixed)
        
        low, high = search_range
        
        # Check if there's a switch in the range
        low_pref = get_preference(low)
        high_pref = get_preference(high)
        
        if low_pref == high_pref or low_pref == 'ERROR' or high_pref == 'ERROR':
            return None
        
        # Binary search
        iterations = 0
        max_iterations = 50
        
        while high - low > tolerance and iterations < max_iterations:
            mid = (low + high) / 2
            mid_pref = get_preference(mid)
            
            if mid_pref == 'ERROR':
                break
            
            if mid_pref == low_pref:
                low = mid
            else:
                high = mid
            
            iterations += 1
        
        return (low + high) / 2
    
    def plot_threshold_trend(self, threshold_data, save_plot=True):
        """
        Plot how switching thresholds change with vehicle count.
        
        Args:
            threshold_data (dict): Data from analyze_switching_threshold_trend
            save_plot (bool): Whether to save the plot
        """
        # Extract data for plotting
        vehicle_counts = []
        thresholds = []
        
        for n_vehicles, data in threshold_data.items():
            if data['switching_point'] is not None:
                vehicle_counts.append(n_vehicles)
                thresholds.append(data['switching_point'])
        
        if not vehicle_counts:
            print("No threshold data to plot.")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot threshold trend
        plt.plot(vehicle_counts, thresholds, 'bo-', linewidth=2, markersize=8, label='Switching Threshold')
        
        # Add trend line
        if len(vehicle_counts) > 2:
            z = np.polyfit(vehicle_counts, thresholds, 1)
            p = np.poly1d(z)
            plt.plot(vehicle_counts, p(vehicle_counts), 'r--', alpha=0.7, label=f'Trend (slope={z[0]:.4f})')
        
        plt.xlabel('Number of Vehicles')
        plt.ylabel(f'Switching Threshold ({list(threshold_data.values())[0]["fixed_param"].replace("p_", "").upper()} Probability)')
        plt.title(f'TCP vs UDP Switching Threshold Trend\n({list(threshold_data.values())[0]["fixed_param"]} = {list(threshold_data.values())[0]["fixed_value"]})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(vehicle_counts)
        
        # Add value annotations
        for i, (x, y) in enumerate(zip(vehicle_counts, thresholds)):
            plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
        
        if save_plot:
            fixed_param = list(threshold_data.values())[0]["fixed_param"]
            fixed_value = list(threshold_data.values())[0]["fixed_value"]
            filename = f'threshold_trend_{fixed_param}_{fixed_value}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Threshold trend plot saved as {filename}")
        
        plt.show()
    
    def analyze_preference_statistics(self):
        """
        Generate detailed statistics about TCP vs UDP preferences across configurations.
        
        Returns:
            dict: Statistical analysis results
        """
        if not self.results_by_config:
            print("No results to analyze. Run analysis first.")
            return {}
        
        print("=== PREFERENCE STATISTICS ANALYSIS ===")
        
        stats = {
            'by_configuration': {},
            'by_message_count': {},
            'overall_trends': {}
        }
        
        # Group by message count and vehicle count
        by_messages = {}
        for config_key, results in self.results_by_config.items():
            n_messages = results['n_messages']
            n_vehicles = results['n_vehicles']
            
            if n_messages not in by_messages:
                by_messages[n_messages] = {}
            
            by_messages[n_messages][n_vehicles] = results
        
        # Calculate statistics by message count
        for n_messages, vehicle_configs in by_messages.items():
            vehicle_counts = sorted(vehicle_configs.keys())
            tcp_ratios = [vehicle_configs[v]['tcp_preference_ratio'] for v in vehicle_counts]
            udp_ratios = [vehicle_configs[v]['udp_preference_ratio'] for v in vehicle_counts]
            
            # Calculate trend for this message count
            if len(vehicle_counts) > 1:
                tcp_trend = np.polyfit(vehicle_counts, tcp_ratios, 1)[0]  # slope
                udp_trend = np.polyfit(vehicle_counts, udp_ratios, 1)[0]  # slope
            else:
                tcp_trend = 0
                udp_trend = 0
            
            stats['by_message_count'][n_messages] = {
                'vehicle_counts': vehicle_counts,
                'tcp_ratios': tcp_ratios,
                'udp_ratios': udp_ratios,
                'tcp_trend_slope': tcp_trend,
                'udp_trend_slope': udp_trend
            }
            
            print(f"{n_messages} message(s) - TCP trend: {tcp_trend:+.4f} per vehicle")
            if len(vehicle_counts) > 1:
                print(f"  TCP preference: {tcp_ratios[0]:.1%} → {tcp_ratios[-1]:.1%}")
                print(f"  UDP preference: {udp_ratios[0]:.1%} → {udp_ratios[-1]:.1%}")
        
        # Store individual configuration stats
        for config_key, results in self.results_by_config.items():
            stats['by_configuration'][config_key] = {
                'tcp_preference_ratio': results['tcp_preference_ratio'],
                'udp_preference_ratio': results['udp_preference_ratio'],
                'equal_ratio': results['equal_ratio'],
                'boundary_count': len(results['switching_boundaries']),
                'analysis_time': results['analysis_time'],
                'n_messages': results['n_messages'],
                'n_vehicles': results['n_vehicles']
            }
        
        return stats


def run_multi_vehicle_analysis():
    """Run the complete multi-vehicle threshold analysis."""
    print("=== MULTI-VEHICLE THRESHOLD ANALYSIS RUNNER ===")
    print("Progressive analysis: 1 message, increasing vehicles")
    print()
    
    # Create analyzer
    analyzer = MultiVehicleThresholdAnalyzer(max_vehicles=6)
    
    # Run progressive analysis
    print("1. Running progressive vehicle analysis...")
    results = analyzer.run_progressive_analysis(
        n_messages=1,
        max_vehicles=6,
        resolution=0.04  # Fine resolution for detailed analysis
    )
    
    # Generate comparison plots
    print("\n2. Generating comparison plots...")
    analyzer.plot_configuration_comparison(save_plot=True)
    
    # Analyze switching threshold trends
    print("\n3. Analyzing switching threshold trends...")
    threshold_data_tcp = analyzer.analyze_switching_threshold_trend(
        n_messages=1,
        max_vehicles=6,
        fixed_param='p_tcp',
        fixed_value=0.8
    )
    
    threshold_data_udp = analyzer.analyze_switching_threshold_trend(
        n_messages=1,
        max_vehicles=6,
        fixed_param='p_udp',
        fixed_value=0.6
    )
    
    # Plot threshold trends
    print("\n4. Plotting threshold trends...")
    analyzer.plot_threshold_trend(threshold_data_tcp, save_plot=True)
    analyzer.plot_threshold_trend(threshold_data_udp, save_plot=True)
    
    # Generate detailed statistics
    print("\n5. Generating detailed statistics...")
    stats = analyzer.analyze_preference_statistics()
    
    # Generate summary report
    print("\n6. Generating summary report...")
    analyzer.generate_summary_report()
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Key findings:")
    
    if 'overall_trends' in stats and stats['overall_trends']:
        tcp_slope = stats['overall_trends']['tcp_trend_slope']
        udp_slope = stats['overall_trends']['udp_trend_slope']
        
        print(f"- TCP preference trend: {tcp_slope:+.4f} per additional vehicle")
        print(f"- UDP preference trend: {udp_slope:+.4f} per additional vehicle")
        
        if tcp_slope > 0.01:
            print("- TCP becomes MORE preferred as vehicle count increases")
        elif tcp_slope < -0.01:
            print("- TCP becomes LESS preferred as vehicle count increases")
        else:
            print("- TCP preference is relatively stable across vehicle counts")
    
    # Print switching threshold insights
    tcp_thresholds = [data['switching_point'] for data in threshold_data_tcp.values() if data['switching_point'] is not None]
    udp_thresholds = [data['switching_point'] for data in threshold_data_udp.values() if data['switching_point'] is not None]
    
    if tcp_thresholds:
        print(f"- TCP switching thresholds range: {min(tcp_thresholds):.3f} to {max(tcp_thresholds):.3f}")
    if udp_thresholds:
        print(f"- UDP switching thresholds range: {min(udp_thresholds):.3f} to {max(udp_thresholds):.3f}")
    
    return analyzer, results, threshold_data_tcp, threshold_data_udp, stats


if __name__ == "__main__":
    # Run the multi-vehicle analysis
    analyzer, results = run_multi_vehicle_analysis()
