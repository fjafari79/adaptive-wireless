import numpy as np
import matplotlib.pyplot as plt
from weighted_graph import *

class SimpleThresholdAnalyzer:
    """
    Analyze the simplest case: 1 message, 1 vehicle, state transition from 1 to 0.
    Find the exact threshold where UDP and TCP preferences switch.
    """
    
    def __init__(self):
        """Initialize the analyzer for the simplest case."""
        self.n_messages = 1
        self.n_vehicles = 1
        self.results = []
        
    def analyze_threshold_sweep(self, p_tcp_range=(0.1, 0.99), p_udp_range=(0.1, 0.99), resolution=0.01):
        """
        Sweep through different probability combinations to find the switching threshold.
        
        Args:
            p_tcp_range (tuple): (min, max) TCP probability range
            p_udp_range (tuple): (min, max) UDP probability range
            resolution (float): Step size for probability sweep
            
        Returns:
            dict: Analysis results including switching points
        """
        print("=== SIMPLE THRESHOLD ANALYSIS ===")
        print("Case: 1 message, 1 vehicle, state transition 1 → 0")
        print(f"TCP probability range: {p_tcp_range}")
        print(f"UDP probability range: {p_udp_range}")
        print(f"Resolution: {resolution}")
        print()
        
        # Generate probability grids
        p_tcp_values = np.arange(p_tcp_range[0], p_tcp_range[1] + resolution, resolution)
        p_udp_values = np.arange(p_udp_range[0], p_udp_range[1] + resolution, resolution)
        
        # Results storage
        tcp_preferred_points = []
        udp_preferred_points = []
        switching_boundaries = []
        
        print("Analyzing probability combinations...")
        total_combinations = len(p_tcp_values) * len(p_udp_values)
        analyzed = 0
        
        for p_tcp in p_tcp_values:
            for p_udp in p_udp_values:
                # Analyze this specific probability combination
                preferred_action = self._analyze_single_combination(p_tcp, p_udp)
                
                if preferred_action == 'TCP':
                    tcp_preferred_points.append((p_tcp, p_udp))
                elif preferred_action == 'UDP':
                    udp_preferred_points.append((p_tcp, p_udp))
                
                analyzed += 1
                if analyzed % 1000 == 0:
                    print(f"  Progress: {analyzed}/{total_combinations} ({100*analyzed/total_combinations:.1f}%)")
        
        # Find switching boundaries
        switching_boundaries = self._find_switching_boundaries(tcp_preferred_points, udp_preferred_points, resolution)
        
        results = {
            'p_tcp_range': p_tcp_range,
            'p_udp_range': p_udp_range,
            'resolution': resolution,
            'tcp_preferred_points': tcp_preferred_points,
            'udp_preferred_points': udp_preferred_points,
            'switching_boundaries': switching_boundaries,
            'total_combinations': total_combinations
        }
        
        self.results = results
        
        print(f"Analysis completed!")
        print(f"TCP preferred combinations: {len(tcp_preferred_points)}")
        print(f"UDP preferred combinations: {len(udp_preferred_points)}")
        print(f"Switching boundary points found: {len(switching_boundaries)}")
        
        return results
    
    def _analyze_single_combination(self, p_tcp, p_udp):
        """
        Analyze a single probability combination for the 1→0 transition.
        
        Args:
            p_tcp (float): TCP success probability
            p_udp (float): UDP success probability
            
        Returns:
            str: 'TCP', 'UDP', or 'EQUAL' based on which action is preferred
        """
        # Calculate weights directly instead of using the graph
        # For the 1->0 transition (delta = 1):
        # UDP weight = 1 + (1/p_udp - 1)^1 = 1/p_udp  
        # TCP weight = 1/p_tcp
        
        try:
            # Calculate UDP weight (delta=1 for 1->0 transition)
            if p_udp == 0:
                udp_weight = float('inf')
            else:
                udp_weight = 1 / p_udp
            
            # Calculate TCP weight
            if p_tcp == 0:
                tcp_weight = float('inf')
            else:
                tcp_weight = 1 / p_tcp
            
            # Determine preferred action based on weights (lower is better)
            if tcp_weight < udp_weight:
                return 'TCP'
            elif udp_weight < tcp_weight:
                return 'UDP'
            else:
                return 'EQUAL'
                
        except Exception as e:
            return 'ERROR'
    
    def _find_switching_boundaries(self, tcp_points, udp_points, resolution):
        """
        Find the boundary points where preferences switch between TCP and UDP.
        
        Args:
            tcp_points (list): List of (p_tcp, p_udp) tuples where TCP is preferred
            udp_points (list): List of (p_tcp, p_udp) tuples where UDP is preferred
            resolution (float): Grid resolution
            
        Returns:
            list: Boundary points where preferences switch
        """
        boundaries = []
        
        # Convert to sets for faster lookup
        tcp_set = set(tcp_points)
        udp_set = set(udp_points)
        
        # Check each TCP point to see if it has UDP neighbors
        for p_tcp, p_udp in tcp_points:
            # Check neighboring points
            neighbors = [
                (p_tcp + resolution, p_udp),
                (p_tcp - resolution, p_udp),
                (p_tcp, p_udp + resolution),
                (p_tcp, p_udp - resolution)
            ]
            
            for neighbor in neighbors:
                if neighbor in udp_set:
                    # Found a boundary point
                    boundaries.append(((p_tcp, p_udp), neighbor))
        
        return boundaries
    
    def plot_threshold_map(self, save_plot=True):
        """
        Create a visual map showing TCP vs UDP preferences across probability space.
        
        Args:
            save_plot (bool): Whether to save the plot to file
        """
        if not self.results:
            print("No results to plot. Run analyze_threshold_sweep() first.")
            return
        
        results = self.results
        
        # Extract data
        tcp_points = np.array(results['tcp_preferred_points'])
        udp_points = np.array(results['udp_preferred_points'])
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        
        # Plot TCP preferred region
        if len(tcp_points) > 0:
            plt.scatter(tcp_points[:, 0], tcp_points[:, 1], 
                       c='blue', alpha=0.6, s=20, label='TCP Preferred', marker='s')
        
        # Plot UDP preferred region
        if len(udp_points) > 0:
            plt.scatter(udp_points[:, 0], udp_points[:, 1], 
                       c='orange', alpha=0.6, s=20, label='UDP Preferred', marker='o')
        
        # Plot switching boundaries
        boundaries = results['switching_boundaries']
        if boundaries:
            boundary_tcp = [b[0] for b in boundaries]
            boundary_udp = [b[1] for b in boundaries]
            
            # Plot boundary points
            if boundary_tcp:
                boundary_tcp = np.array(boundary_tcp)
                plt.scatter(boundary_tcp[:, 0], boundary_tcp[:, 1], 
                           c='red', s=50, label='Boundary (TCP side)', marker='x')
            
            if boundary_udp:
                boundary_udp = np.array(boundary_udp)
                plt.scatter(boundary_udp[:, 0], boundary_udp[:, 1], 
                           c='darkred', s=50, label='Boundary (UDP side)', marker='+')
        
        # Add diagonal reference line (p_tcp = p_udp)
        min_p = min(results['p_tcp_range'][0], results['p_udp_range'][0])
        max_p = max(results['p_tcp_range'][1], results['p_udp_range'][1])
        plt.plot([min_p, max_p], [min_p, max_p], 'k--', alpha=0.5, label='p_tcp = p_udp')
        
        plt.xlabel('TCP Success Probability (p_tcp)')
        plt.ylabel('UDP Success Probability (p_udp)')
        plt.title('TCP vs UDP Preference Map\n(1 message, 1 vehicle, 1→0 transition)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(results['p_tcp_range'])
        plt.ylim(results['p_udp_range'])
        
        # Add text annotation with analysis details
        plt.text(0.02, 0.98, f"Resolution: {results['resolution']}\n"
                             f"Total combinations: {results['total_combinations']}\n"
                             f"TCP preferred: {len(tcp_points)}\n"
                             f"UDP preferred: {len(udp_points)}", 
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plot:
            filename = f'simple_threshold_map_res_{results["resolution"]}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Threshold map saved as {filename}")
        
        plt.show()
    
    def analyze_specific_threshold(self, p_tcp, p_udp):
        """
        Analyze a specific probability combination in detail.
        
        Args:
            p_tcp (float): TCP success probability
            p_udp (float): UDP success probability
            
        Returns:
            dict: Detailed analysis results
        """
        print(f"=== DETAILED ANALYSIS FOR p_tcp={p_tcp}, p_udp={p_udp} ===")
        
        try:
            # Calculate weights directly for the 1->0 transition (delta=1)
            # UDP weight = 1 + (1/p_udp - 1)^1 = 1/p_udp  
            # TCP weight = 1/p_tcp
            
            if p_udp == 0:
                udp_weight = float('inf')
            else:
                udp_weight = 1 / p_udp
            
            if p_tcp == 0:
                tcp_weight = float('inf')
            else:
                tcp_weight = 1 / p_tcp
            
            # Create actions analysis
            actions_analysis = {
                'TCP': {
                    'weight': tcp_weight,
                    'action_type': 0,
                    'destination_matrix': np.zeros((self.n_messages, self.n_vehicles), dtype=int)
                },
                'UDP': {
                    'weight': udp_weight,
                    'action_type': 1,
                    'destination_matrix': np.zeros((self.n_messages, self.n_vehicles), dtype=int)
                }
            }
            
            print(f"TCP action:")
            print(f"  Weight (cost): {tcp_weight:.6f}")
            print(f"  Destination: [0]")
            print(f"  Probability of success: {p_tcp}")
            print()
            
            print(f"UDP action:")
            print(f"  Weight (cost): {udp_weight:.6f}")
            print(f"  Destination: [0]")
            print(f"  Probability of success: {p_udp}")
            print()
            
            # Determine preferred action
            if tcp_weight < udp_weight:
                preferred = 'TCP'
                weight_difference = udp_weight - tcp_weight
            elif udp_weight < tcp_weight:
                preferred = 'UDP'
                weight_difference = tcp_weight - udp_weight
            else:
                preferred = 'EQUAL'
                weight_difference = 0
            
            print(f"Preferred action: {preferred}")
            if weight_difference > 0:
                print(f"Weight advantage: {weight_difference:.6f}")
            
            return {
                'p_tcp': p_tcp,
                'p_udp': p_udp,
                'actions_analysis': actions_analysis,
                'preferred_action': preferred,
                'weight_difference': weight_difference,
                'tcp_weight': tcp_weight,
                'udp_weight': udp_weight
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def find_exact_switching_point(self, fixed_param='p_tcp', fixed_value=0.8, 
                                  search_range=(0.1, 0.99), tolerance=1e-6):
        """
        Find the exact switching point by fixing one parameter and varying the other.
        
        Args:
            fixed_param (str): 'p_tcp' or 'p_udp' - which parameter to fix
            fixed_value (float): Value of the fixed parameter
            search_range (tuple): (min, max) range for the varying parameter
            tolerance (float): Tolerance for binary search convergence
            
        Returns:
            dict: Results including the exact switching point
        """
        print(f"=== FINDING EXACT SWITCHING POINT ===")
        print(f"Fixed parameter: {fixed_param} = {fixed_value}")
        print(f"Search range: {search_range}")
        print(f"Tolerance: {tolerance}")
        print()
        
        def get_preference(varying_value):
            """Helper function to get preference for given parameter values."""
            if fixed_param == 'p_tcp':
                return self._analyze_single_combination(fixed_value, varying_value)
            else:
                return self._analyze_single_combination(varying_value, fixed_value)
        
        # Binary search for the switching point
        low, high = search_range
        
        # Check if there's actually a switch in the range
        low_pref = get_preference(low)
        high_pref = get_preference(high)
        
        print(f"Preference at {search_range[0]}: {low_pref}")
        print(f"Preference at {search_range[1]}: {high_pref}")
        
        if low_pref == high_pref:
            print("No switching point found in the given range.")
            return {
                'fixed_param': fixed_param,
                'fixed_value': fixed_value,
                'switching_point': None,
                'message': 'No switch in range'
            }
        
        # Binary search
        iterations = 0
        max_iterations = 100
        
        while high - low > tolerance and iterations < max_iterations:
            mid = (low + high) / 2
            mid_pref = get_preference(mid)
            
            print(f"Iteration {iterations + 1}: {mid:.6f} -> {mid_pref}")
            
            if mid_pref == low_pref:
                low = mid
            else:
                high = mid
            
            iterations += 1
        
        switching_point = (low + high) / 2
        
        print(f"\nSwitching point found: {switching_point:.6f}")
        print(f"Converged after {iterations} iterations")
        
        # Verify the switching point
        before_pref = get_preference(switching_point - tolerance)
        after_pref = get_preference(switching_point + tolerance)
        
        print(f"Preference just before: {before_pref}")
        print(f"Preference just after: {after_pref}")
        
        return {
            'fixed_param': fixed_param,
            'fixed_value': fixed_value,
            'switching_point': switching_point,
            'iterations': iterations,
            'tolerance_achieved': high - low,
            'preference_before': before_pref,
            'preference_after': after_pref
        }


def run_simple_analysis():
    """Run the complete simple threshold analysis."""
    print("=== SIMPLE THRESHOLD ANALYSIS RUNNER ===")
    print("Analyzing the simplest case: 1 message, 1 vehicle, 1→0 transition")
    print()
    
    analyzer = SimpleThresholdAnalyzer()
    
    # 1. Quick threshold map with coarse resolution
    print("1. Creating threshold map...")
    results = analyzer.analyze_threshold_sweep(
        p_tcp_range=(0.1, 0.99), 
        p_udp_range=(0.1, 0.99), 
        resolution=0.02  # Coarse resolution for quick overview
    )
    
    analyzer.plot_threshold_map(save_plot=True)
    print()
    
    # 2. Analyze specific interesting points
    print("2. Analyzing specific probability combinations...")
    test_points = [
        (0.5, 0.5),   # Equal probabilities
        (0.8, 0.6),   # High TCP, medium UDP
        (0.6, 0.8),   # Medium TCP, high UDP
        (0.9, 0.5),   # Very high TCP, medium UDP
        (0.5, 0.9),   # Medium TCP, very high UDP
    ]
    
    for p_tcp, p_udp in test_points:
        print(f"\n--- Analyzing p_tcp={p_tcp}, p_udp={p_udp} ---")
        analysis = analyzer.analyze_specific_threshold(p_tcp, p_udp)
        print()
    
    # 3. Find exact switching points
    print("3. Finding exact switching points...")
    
    # Fix p_tcp=0.8, find switching point in p_udp
    switching_result1 = analyzer.find_exact_switching_point(
        fixed_param='p_tcp', 
        fixed_value=0.8, 
        search_range=(0.1, 0.99)
    )
    print()
    
    # Fix p_udp=0.6, find switching point in p_tcp
    switching_result2 = analyzer.find_exact_switching_point(
        fixed_param='p_udp', 
        fixed_value=0.6, 
        search_range=(0.1, 0.99)
    )
    print()
    
    print("=== ANALYSIS COMPLETE ===")
    print("Results:")
    print(f"- Threshold map created with {results['total_combinations']} combinations")
    print(f"- TCP preferred: {len(results['tcp_preferred_points'])} combinations")
    print(f"- UDP preferred: {len(results['udp_preferred_points'])} combinations")
    if switching_result1['switching_point']:
        print(f"- Switching point (p_tcp=0.8): p_udp = {switching_result1['switching_point']:.6f}")
    if switching_result2['switching_point']:
        print(f"- Switching point (p_udp=0.6): p_tcp = {switching_result2['switching_point']:.6f}")
    
    return analyzer, results, switching_result1, switching_result2


if __name__ == "__main__":
    # Run the simple analysis
    analyzer, results, switch1, switch2 = run_simple_analysis()
