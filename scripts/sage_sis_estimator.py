#!/usr/bin/env python
"""
SIS parameter estimator using the lattice-estimator library.
Usage: python sage_sis_estimator.py <n> <m> <q> <length_bound> <norm>
"""

import sys
import math

# Add lattice-estimator to path if needed
sys.path.insert(0, 'lattice-estimator')

from estimator import *
from sage.all import oo

def estimate_sis(n, m, q, length_bound, norm):
    """
    Estimate the security of SIS parameters using lattice-estimator.
    
    Args:
        n: Dimension
        m: Number of samples
        q: Modulus
        length_bound: Maximum length of the solution
        norm: Either 2 (L2 norm) or "oo" (infinity norm)
    
    Returns:
        Log2 of the number of ring operations (ROP)
    """
    # Convert norm to the appropriate value
    if norm == "2":
        norm_value = 2
    elif norm == "oo":
        norm_value = oo
    else:
        raise ValueError(f"Invalid norm: {norm}. Must be '2' or 'oo'")
    
    # Create SIS parameters
    params = SIS.Parameters(
        n=n,
        m=m,
        q=q,
        length_bound=length_bound,
        norm=norm_value
    )
    
    # Run the lattice attack estimator
    result = SIS.lattice(params)
    
    # Extract ROP (ring operations) and compute log2
    rop = result["rop"]
    log2_rop = math.log2(float(rop))
    
    return log2_rop

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print(f"Usage: {sys.argv[0]} <n> <m> <q> <length_bound> <norm>", file=sys.stderr)
        print(f"  norm should be either '2' or 'oo'", file=sys.stderr)
        sys.exit(1)
    
    try:
        n = int(sys.argv[1])
        m = int(sys.argv[2])
        q = int(sys.argv[3])
        length_bound = int(sys.argv[4])
        norm = sys.argv[5]
        
        log2_rop = estimate_sis(n, m, q, length_bound, norm)
        
        # Print only the result (for parsing by Rust)
        print(log2_rop)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
