import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def parse_file(filename):
    data = defaultdict(lambda: defaultdict(list))
    
    current_nt = None

    # Strict nt pattern: must start with dashes and end with dashes
    nt_pattern = re.compile(r"^-{14}\s*nt\s+(\d+)")

    patterns = {
        "time_total": re.compile(r"Time total time stepping:\s*([0-9eE\.\-\+]+)"),
        "time_vel": re.compile(r"Time total velocity:\s*([0-9eE\.\-\+]+)"),
        "time_scheme": re.compile(r"Time total scheme:\s*([0-9eE\.\-\+]+)"),
        "total_iters": re.compile(r"Total number of iterations over all times:\s*([0-9eE\.\-\+]+)"),
        "l2norm": re.compile(r"l2norm\s*=\s*([0-9eE\.\-\+]+)"),
        "tracer_minmax": re.compile(
            r"Min and max tracer at final time step:\s*\(np\.float64\(([^)]+)\),\s*np\.float64\(([^)]+)\)\)"
        ),
        "C_minmax": re.compile(
            r"Minimum and maximum C over all times:\s*([0-9eE\.\-\+]+),\s*([0-9eE\.\-\+]+)"
        ),
    }

    with open(filename, "r") as f:
        for line in f:
            
            # Strict nt detection
            nt_match = nt_pattern.match(line.strip())
            if nt_match:
                current_nt = int(nt_match.group(1))
                continue

            if current_nt is None:
                continue
            
            # Scalar variables
            for key in ["time_total", "time_vel", "time_scheme", "total_iters", "l2norm"]:
                match = patterns[key].search(line)
                if match:
                    data[current_nt][key].append(float(match.group(1)))
                    break
            
            # Tracer min/max
            match = patterns["tracer_minmax"].search(line)
            if match:
                data[current_nt]["tracer_min"].append(float(match.group(1)))
                data[current_nt]["tracer_max"].append(float(match.group(2)))
                continue
            
            # C min/max
            match = patterns["C_minmax"].search(line)
            if match:
                data[current_nt]["C_min"].append(float(match.group(1)))
                data[current_nt]["C_max"].append(float(match.group(2)))
                continue

    return data


def turn_into_np_arrays(data):
    for nt, values in data.items():
        for key, val_list in values.items():
            if key in ['time_total', 'time_vel', 'time_scheme']:
                data[nt][key] = np.array(val_list)
            else:
                data[nt][key] = np.array(val_list[0])
    return data


def add_mean_timings(data):
    for nt, values in data.items():
        for key in ['time_total', 'time_vel', 'time_scheme']:
            if key in values:
                data[nt][f"{key}_mean"] = np.mean(values[key])
    return data


# ---- Secondary x-axis (C) ----
# Build mapping using interpolation (robust even if not perfectly linear)
def dt_to_C(x, dt, C):
    return np.interp(x, dt[::-1], C[::-1])  # reversed for increasing order


def C_to_dt(x, dt, C):
    return np.interp(x, C[::-1], dt[::-1])


def plot_timings(data):
    nts = sorted(data.keys())
    
    # Extract data
    dt = np.array([data[nt]["dt"] for nt in nts])
    C  = np.array([data[nt]["C"]  for nt in nts])
    
    l2 = np.array([data[nt]["l2norm"] for nt in nts])
    time_scheme = np.array([np.mean(data[nt]["time_scheme"]) for nt in nts])
    total_iters = np.array([data[nt]["total_iters"] for nt in nts])
    
    time_per_step = time_scheme / np.array(nts)
    
    # ---- Plot ----
    fig, axs = plt.subplots(4, 1, figsize=(7, 10), sharex=True)
   
    # 1) l2 norm
    axs[0].plot(dt, l2, marker='o')
    axs[0].set_ylabel("l2 norm")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].grid(True, which="both", ls="--", alpha=0.5)
    
    # 2) total scheme time
    axs[1].plot(dt, time_scheme, marker='o')
    axs[1].set_ylabel("time (scheme)")
    axs[1].set_xscale("log")
    axs[1].grid(True, which="both", ls="--", alpha=0.5)
    
    # 3) time per step
    axs[2].plot(dt, time_per_step, marker='o')
    axs[2].set_ylabel("time / nt")
    axs[2].set_xscale("log")
    axs[2].grid(True, which="both", ls="--", alpha=0.5)
    
    # 4) total iterations
    axs[3].plot(dt, total_iters, marker='o')
    axs[3].set_ylabel("total iterations")
    axs[3].set_xlabel("dt")
    axs[3].set_xscale("log")
    axs[3].grid(True, which="both", ls="--", alpha=0.5)

    
    #secax = axs[-1].secondary_xaxis('top', functions=(dt_to_C, C_to_dt))
    #secax.set_xlabel("C")

    # add cmax and cmin x axes
    
    plt.tight_layout()
    plt.show()



def main():
    filename = "singlerun_test_accuracy_timings-fourthversion-k3.txt" # "singlerun_test_accuracy_timings-fourthversion.txt" #"output_run_20x-fourthversion.txt"
    data = parse_file(filename)
    data = turn_into_np_arrays(data)
    data = add_mean_timings(data)

    # add dt list
    simulated_time = 100.0
    for nt, values in data.items():
        data[nt]["dt"] = simulated_time / nt

    #for nt, values in data.items():
    #    print(f"nt: {nt}")
    #    for key, val_list in values.items():
    #        print(f"  {key}: {val_list}")
    #    print()

    for nt, values in data.items():
        print(f"nt: {nt}, dt: {values['dt']:.2e}, C_min: {values['C_min']:.2e}, C_max: {values['C_max']:.2e}, l2norm: {values['l2norm']:.2e}, time_scheme_mean: {values['time_scheme_mean']:.2f}s, total_iters: {values['total_iters']}")

    plot_timings(data)

if __name__ == "__main__":
    main()