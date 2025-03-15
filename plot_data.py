#!/usr/bin/env python3
"""
plot_evaluation.py

This script reads an evaluation log file (e.g. evaluation_twins.txt or evaluation_ihdp.txt)
with a format such as:

    === Comprehensive Evaluation for dataset: twins ===

    --- Train shifted = False ---

    *** Experiment: flipping probability p = 0.0 ***
    Loaded Twins Z from data/TWINS/processed_z_p0.0.csv
    [DEBUG] EPOCH 0
    Epoch 0 loss: 14881.5327
    ...
    Cross-validation on Test Set Evaluation:
    CEVAE Test CV (Twins): {'ATE_mean': -0.0021141844, 'ATE_std': 0.0031912758, 'PEHE_mean': 0.554667, 'PEHE_std': 0.007526595, 'ATE_Abs_Error_mean': 0.015048586, 'ATE_Abs_Error_std': 0.005798056}

    True causal effects on the test set:
    True ATE (test): -0.025459098
    True ITE (test) snippet: [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

    Running baseline evaluations using utility models via 3-fold CV...
    IPW CV (Twins): ATE=-0.0509, ATE_Abs_Error=0.0254
    DML CV (Twins): ATE=0.0003, PEHE=0.3237, ATE_Abs_Error=0.0254
    X-Learner CV (Twins): ATE=0.0046, PEHE=0.3411, ATE_Abs_Error=0.0297
    SVM CV (Twins): ATE=0.0000, PEHE=0.3218, ATE_Abs_Error=0.0251
    KNN CV (Twins): ATE=-0.0000, PEHE=0.3385, ATE_Abs_Error=0.0251
    Interacted LR CV (Twins): ATE=0.0059, PEHE=0.7833, ATE_Abs_Error=0.0311
    XGBoost CV (Twins): ATE=0.0014, PEHE=0.3255, ATE_Abs_Error=0.0265

For the IPW baseline, if the “ATE_Abs_Error” is not provided, it is computed as
    |(IPW ATE) – (True ATE (test))|
using the “True ATE (test)” value given above.

Then, for each metric (Absolute ATE Error and PEHE) the script produces two plots:
  • One for “Train shifted = False”
  • One for “Train shifted = True”

In each plot, the x‐axis is the flipping probability p (from 0.0 to 0.5),
and one line is plotted for each model (CEVAE, IPW, DML, X-Learner, SVM, KNN, Interacted LR, XGBoost).
For fairness, the y-axis limits for a given metric are determined by the overall maximum
value (across both shift conditions) for that metric.
Finally, the legend is plotted with a smaller font size so that it does not block the data.
All four plots are saved in the folder “evaluation_plots”.
"""

import re
import os
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(filename):
    """
    Parses the log file and returns a nested dictionary:
    
    results[shift_state][p] = {
         "true_ate": float,
         "CEVAE": {"ATE_Abs_Error": float, "PEHE": float},
         "baseline": {
             model_name: {"ATE": float, "ATE_Abs_Error": float, "PEHE": float},
             ...
         }
    }
    shift_state is a string: "False" or "True".
    """
    results = {"False": {}, "True": {}}
    current_shift = None
    current_p = None
    current_true_ate = None

    # Regular expressions
    shift_re = re.compile(r"^--- Train shifted = (True|False) ---")
    exp_re = re.compile(r"^\*\*\* Experiment: flipping probability p = ([\d\.]+) \*\*\*")
    true_ate_re = re.compile(r"^True ATE \(test\):\s*([-\d\.]+)")
    cevae_re = re.compile(r"CEVAE Test CV \([^)]+\):\s*(\{[^}]+\})")
    baseline_re = re.compile(r"^(IPW|DML|X-Learner|SVM|KNN|Interacted LR|XGBoost) CV \([^)]+\):\s*(.+)$")
    
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            # Check for shift state header.
            m = shift_re.match(line)
            if m:
                current_shift = m.group(1)
                continue
            # Check for experiment header.
            m = exp_re.match(line)
            if m:
                current_p = float(m.group(1))
                results[current_shift][current_p] = {"CEVAE": {}, "baseline": {}}
                current_true_ate = None  # reset
                continue
            # Check for True ATE line.
            m = true_ate_re.match(line)
            if m:
                current_true_ate = float(m.group(1))
                if current_p is not None:
                    results[current_shift][current_p]["true_ate"] = current_true_ate
                continue
            # Check for CEVAE line.
            m = cevae_re.search(line)
            if m and current_p is not None:
                try:
                    cevae_dict = eval(m.group(1))
                except Exception:
                    cevae_dict = {}
                results[current_shift][current_p]["CEVAE"]["ATE_Abs_Error"] = cevae_dict.get("ATE_Abs_Error_mean", np.nan)
                results[current_shift][current_p]["CEVAE"]["PEHE"] = cevae_dict.get("PEHE_mean", np.nan)
                continue
            # Check for baseline lines.
            m = baseline_re.match(line)
            if m and current_p is not None:
                model_name = m.group(1)
                metric_str = m.group(2)
                # Parse ATE value.
                ate_match = re.search(r"ATE=([-\d\.]+)", metric_str)
                ate_val = float(ate_match.group(1)) if ate_match else np.nan
                # For IPW, if no ATE_Abs_Error is provided, compute it.
                ate_abs_match = re.search(r"ATE_Abs_Error=([-\d\.]+)", metric_str)
                if model_name == "IPW":
                    if current_true_ate is not None:
                        abs_error = abs(ate_val - current_true_ate)
                    else:
                        abs_error = np.nan
                else:
                    abs_error = float(ate_abs_match.group(1)) if ate_abs_match else np.nan
                # Parse PEHE if available.
                pehe_match = re.search(r"PEHE=([-\d\.]+)", metric_str)
                pehe_val = float(pehe_match.group(1)) if pehe_match else np.nan

                results[current_shift][current_p]["baseline"].setdefault(model_name, {})["ATE"] = ate_val
                results[current_shift][current_p]["baseline"].setdefault(model_name, {})["ATE_Abs_Error"] = abs_error
                results[current_shift][current_p]["baseline"].setdefault(model_name, {})["PEHE"] = pehe_val
    return results

def gather_metric(results, metric):
    """
    Returns a dictionary:
       data[shift_state][p][model] = value
    for the given metric (e.g. "ATE_Abs_Error" or "PEHE") for both CEVAE and each baseline.
    """
    data = {"False": {}, "True": {}}
    for shift in results:
        for p in results[shift]:
            if p not in data[shift]:
                data[shift][p] = {}
            data[shift][p]["CEVAE"] = results[shift][p]["CEVAE"].get(metric, np.nan)
            for model, vals in results[shift][p]["baseline"].items():
                data[shift][p][model] = vals.get(metric, np.nan)
    return data

def overall_max(metric_data, model_list):
    """
    Returns the maximum value across both shift states and all p values for the given metric,
    considering only the specified models.
    """
    max_val = 0
    for shift in metric_data:
        for p in metric_data[shift]:
            for model in model_list:
                val = metric_data[shift][p].get(model, np.nan)
                if not np.isnan(val) and val > max_val:
                    max_val = val
    return max_val if max_val > 0 else 1

def plot_metric(metric_data, metric_label, overall_max_val, shift_state, model_list, output_folder):
    """
    Plots the given metric vs. p for a given shift_state.
    The x-axis shows the flipping probability (p) sorted in ascending order.
    The y-axis is fixed from 0 to overall_max_val*1.1.
    A legend is included with a smaller font size.
    The plot is saved to output_folder with a filename that includes the shift state and metric.
    """
    p_vals = sorted(metric_data[shift_state].keys())
    plt.figure(figsize=(8,6))
    for model in model_list:
        y_vals = [metric_data[shift_state][p].get(model, np.nan) for p in p_vals]
        plt.plot(p_vals, y_vals, marker='o', label=model)
    plt.title(f"{metric_label} vs. Flipping Probability on IHDP Dataset (shifted = {shift_state})")
    plt.xlabel("Flipping probability p")
    plt.ylabel(metric_label)
    plt.ylim(0, overall_max_val * 1.1)
    plt.legend(prop={'size': 8}, loc='best')
    plt.grid(True)
    fname = os.path.join(output_folder, f"{shift_state}_{metric_label.replace(' ', '_')}_vs_p.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved plot: {fname}")

def main():
    # Change this filename as needed (for twins or ihdp)
    log_file = "formatted_twins.txt"  # or "evaluation_ihdp.txt"
    results = parse_log_file(log_file)
    
    # Define the order of models; note that for baseline IPW we use the computed ATE_Abs_Error.
    models1 = ["CEVAE", "IPW", "DML", "X-Learner", "SVM", "Interacted LR", "XGBoost"]
    models2 = ["CEVAE", "DML", "X-Learner", "SVM", "Interacted LR", "XGBoost"]

    ate_data = gather_metric(results, "ATE_Abs_Error")
    print(ate_data)
    pehe_data = gather_metric(results, "PEHE")
    print(pehe_data)

    max_ate = overall_max(ate_data, models1)
    max_pehe = overall_max(pehe_data, models2)
    
    # Create output folder.
    output_folder = "evaluation_plots_twins"
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate two plots for each metric (one per shift condition).
    for shift in ["False", "True"]:
        plot_metric(ate_data, "Absolute ATE Error", max_ate, shift, models1, output_folder)
        plot_metric(pehe_data, "PEHE", max_pehe, shift, models2, output_folder)

if __name__ == "__main__":
    main()
