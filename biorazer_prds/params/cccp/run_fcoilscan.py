#!/usr/bin/env python3
"""
Command line interface for fcoilscan function.
"""

import argparse
import sys
import os
from ..core.fcoilscan import fcoilscan


def main():
    """Main command line interface for fcoilscan."""
    parser = argparse.ArgumentParser(
        description='Local Crick parameter analysis and accommodation index calculation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('input', help='Input structure file (PDB or XYZ)')
    parser.add_argument('--chains', '-c', type=int, required=True,
                       help='Number of chains in the structure')
    parser.add_argument('--window', '-w', type=int, default=7,
                       help='Window size for local analysis')
    parser.add_argument('--ph-ideal', '-p', type=float, default=102.857,
                       help='Ideal phase change in degrees (360/3.5 for 7/2 coils)')
    parser.add_argument('--coor-type', '-t', type=int, default=1,
                       help='Coordinate type (1=PDB, 0=XYZ)')
    parser.add_argument('--output', '-o', default='fcoilscan_results',
                       help='Output directory')
    parser.add_argument('--excel', '-x', action='store_true',
                       help='Generate Excel output file')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Run fcoilscan
        print(f"Performing local analysis on {args.input}")
        print(f"Number of chains: {args.chains}")
        print(f"Window size: {args.window}")
        print(f"Ideal phase change: {args.ph_ideal}°")
        print(f"Output directory: {args.output}")
        
        coil_fits, parameters, ai_phases = fcoilscan(
            args.input,
            args.coor_type,
            args.chains,
            args.window,
            args.ph_ideal,
            args.output,
            args.excel
        )
        
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {args.output}/")
        
        # Display summary
        if coil_fits and 'data' in coil_fits:
            print(f"\nAccommodation Index Summary:")
            for i, row in enumerate(coil_fits['data']):
                chain_id = row[0]
                ai_value = row[3]
                if not (ai_value != ai_value):  # Check for NaN
                    print(f"Chain {chain_id}: AI = {ai_value:.3f}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
