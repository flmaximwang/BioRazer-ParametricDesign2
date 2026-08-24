#!/usr/bin/env python3
"""
Command line interface for fcrick function.
"""

import argparse
import sys
import os
from ..core.fcrick import fcrick


def main():
    """Main command line interface for fcrick."""
    parser = argparse.ArgumentParser(
        description='Fit Crick parameters to protein structure',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('input', help='Input structure file (PDB or XYZ)')
    parser.add_argument('--chains', '-c', type=int, required=True,
                       help='Number of chains in the structure')
    parser.add_argument('--par-type', '-p', default='GENERAL',
                       choices=['GENERAL', 'SYMMETRIC', 'ZOFF-SYMM', 'DPH0-SYMM',
                               'GENERAL-HLXPH', 'SYMMETRIC-HLXPH', 
                               'ZOFF-SYMM-HLXPH', 'DPH0-SYMM-HLXPH'],
                       help='Parameterization type')
    parser.add_argument('--coor-type', '-t', type=int, default=1,
                       help='Coordinate type (1=PDB, 0=XYZ)')
    parser.add_argument('--output', '-o', default='fcrick_results',
                       help='Output file prefix')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output with plots')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Determine output type
        out_type = 1 if args.verbose else args.output
        
        # Run fcrick
        print(f"Fitting Crick parameters to {args.input}")
        print(f"Number of chains: {args.chains}")
        print(f"Parameterization: {args.par_type}")
        
        error, coords, params = fcrick(
            args.input,
            args.chains,
            args.par_type,
            args.coor_type,
            out_type
        )
        
        print(f"\nFitting completed successfully!")
        print(f"RMSD: {error:.6f} Å")
        
        if isinstance(out_type, str):
            print(f"Results saved to: {out_type}.par")
        
    except Exception as e:
        print(f"Error during fitting: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
