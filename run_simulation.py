#!/usr/bin/env python3
"""
PySimulator - Vehicle Communication Network Analysis
Main entry point for running different analysis modes.

Usage:
    python run_simulation.py --mode simulation    # Run batch simulation
    python run_simulation.py --mode gui          # Launch interactive GUI
    python run_simulation.py --mode analysis     # Run analytical tools
    python run_simulation.py --help              # Show help
"""

import sys
import argparse
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def run_simulation_mode():
    """Run the main simulation suite."""
    try:
        sys.path.append(str(Path(__file__).parent / "simulation_suite"))
        from comm_sim import main as run_comm_sim
        print("Starting comprehensive simulation suite...")
        run_comm_sim()
    except ImportError as e:
        print(f"Error importing simulation suite: {e}")
        print("Make sure the repository structure is properly organized.")
        return False
    except Exception as e:
        print(f"Error running simulation: {e}")
        return False
    return True

def run_gui_mode():
    """Launch the interactive GUI."""
    try:
        from gui.main_gui import run_interactive_analysis
        print("Launching interactive GUI...")
        run_interactive_analysis()
    except ImportError as e:
        print(f"Error importing GUI components: {e}")
        print("Make sure the GUI files are in src/gui/")
        return False
    except Exception as e:
        print(f"Error running GUI: {e}")
        return False
    return True

def run_analysis_mode():
    """Run analytical tools."""
    try:
        sys.path.append(str(Path(__file__).parent / "analysis_tools"))
        from optimal_combination_analyzer import main as run_analysis
        print("Starting analytical optimization...")
        run_analysis()
    except ImportError as e:
        print(f"Error importing analysis tools: {e}")
        print("Make sure analysis tools are properly organized.")
        return False
    except Exception as e:
        print(f"Error running analysis: {e}")
        return False
    return True

def show_project_info():
    """Display project information."""
    info = """
PySimulator - Vehicle Communication Network Analysis
==================================================

This project simulates and analyzes communication strategies in vehicle networks,
comparing three approaches:

1. Graph-based Shortest Path Strategy
2. Markov Decision Process (MDP) Strategy  
3. Estimated Heuristic Strategy

Each strategy decides whether to use TCP (reliable unicast) or UDP (unreliable 
broadcast) to efficiently deliver messages to all vehicles.

Key Features:
- Comprehensive simulation with statistical analysis
- Interactive GUI for state exploration
- Advanced visualizations (histograms, CDFs, heatmaps)
- Analytical boundary detection
- CSV export of detailed results

Repository Structure:
- src/                  # Core source code
- simulation_suite/     # Main simulation engine
- analysis_tools/       # Analytical optimization tools
- experiments/          # Research experiments
- tests/               # Unit tests
- docs/                # Documentation
- data/                # Input/output data
"""
    print(info)

def main():
    """Main entry point with command line argument parsing."""
    
    parser = argparse.ArgumentParser(
        description="PySimulator - Vehicle Communication Network Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode simulation    Run comprehensive simulation
  %(prog)s --mode gui          Launch interactive GUI
  %(prog)s --mode analysis     Run analytical optimization
  %(prog)s --info              Show project information
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["simulation", "gui", "analysis"], 
        default="simulation",
        help="Operation mode (default: simulation)"
    )
    
    parser.add_argument(
        "--info", 
        action="store_true",
        help="Show project information"
    )
    
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Test repository structure"
    )
    
    args = parser.parse_args()
    
    if args.info:
        show_project_info()
        return 0
    
    if args.test:
        print("Running repository structure test...")
        try:
            from test_structure import main as test_main
            return test_main()
        except ImportError:
            print("test_structure.py not found")
            return 1
    
    print("PySimulator - Vehicle Communication Network Analysis")
    print("=" * 55)
    
    # Run the selected mode
    success = False
    if args.mode == "simulation":
        success = run_simulation_mode()
    elif args.mode == "gui":
        success = run_gui_mode()
    elif args.mode == "analysis":
        success = run_analysis_mode()
    
    if success:
        print("\n✅ Execution completed successfully!")
        return 0
    else:
        print("\n❌ Execution failed. Check error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
