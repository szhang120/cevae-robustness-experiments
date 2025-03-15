#!/usr/bin/env python3
import sys
import re

# Regex for extracting the True ATE from lines like:
#   True ATE (test): -0.025459098
true_ate_pattern = re.compile(r'^True ATE \(test\):\s*([-]?\d*\.?\d+(?:[eE][-+]?\d+)?)')

# Regex for extracting IPW ATE from lines like:
#   IPW CV (ATE) for Twins: -0.0509
ipw_pattern = re.compile(r'^IPW CV \(ATE\) for IHDP:\s*([-]?\d*\.?\d+(?:[eE][-+]?\d+)?)')

current_true_ate = None  # Will store the most recently seen True ATE

for line in sys.stdin:
    line = line.rstrip('\n')  # Strip trailing newline for processing

    # 1) Check if this line updates our current True ATE
    m_true = true_ate_pattern.match(line)
    if m_true:
        current_true_ate = float(m_true.group(1))
        # Output this line exactly as-is
        print(line)
        continue

    # 2) Check if this line is an IPW CV line we want to augment
    m_ipw = ipw_pattern.match(line)
    if m_ipw:
        ipw_val = float(m_ipw.group(1))
        if current_true_ate is not None:
            ate_abs_error = abs(ipw_val - current_true_ate)
            # Format to match typical 4-decimal style seen in the log:
            new_line = (
                f"IPW CV (IHDP): ATE={ipw_val:.4f}, "
                f"ATE_Abs_Error={ate_abs_error:.4f}"
            )
            print(new_line)
        else:
            # If for some reason we haven't seen a True ATE yet, just pass original through
            print(line)
        continue

    # Otherwise, print the line unchanged
    print(line)
