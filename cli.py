"""
Command-line interface for Junqi Engine.
Unified entry point for all commands: test, gui, calc, train, etc.
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Junqi Engine - Chinese Chess Reversi AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m junqi test          Run all tests
  python -m junqi gui           Start GUI interface  
  python -m junqi calc          Open calculator
  python -m junqi train         Start RL training
  python -m junqi benchmark     Run 50-puzzle suite
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Test command
    subparsers.add_parser('test', help='Run all unit tests')
    
    # GUI command
    subparsers.add_parser('gui', help='Start graphical user interface')
    
    # Calculator command
    subparsers.add_parser('calc', help='Open interactive position calculator')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Start RL training')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    train_parser.add_argument('--games', type=int, default=24, help='Games per epoch')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmark suite')
    benchmark_parser.add_argument('--model', type=str, help='Model file to evaluate')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    # Execute command
    try:
        if args.command == 'test':
            run_tests(args)
        elif args.command == 'gui':
            from junqi.gui import launch_gui
            launch_gui()
        elif args.command == 'calc':
            from ui.calculator import main as calc_main
            calc_main()
        elif args.command == 'train':
            from train.train_rl import main as train_main
            train_main(args)
        elif args.command == 'benchmark':
            from eval.benchmark import main as benchmark_main
            benchmark_main(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def run_tests(args):
    """Run pytest on the tests directory."""
    import subprocess
    result = subprocess.run([
        sys.executable, '-m', 'pytest', 
        'tests/', '-v', '--tb=short'
    ])
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
