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
    Cmax  = np.array([data[nt]["C_max"]  for nt in nts])
    
    l2 = np.array([data[nt]["l2norm"] for nt in nts])
    time_scheme = np.array([np.mean(data[nt]["time_scheme"]) for nt in nts])
    total_iters = np.array([data[nt]["total_iters"] for nt in nts])
    
    time_per_step = time_scheme / np.array(nts)
    iterations_per_step = total_iters / np.array(nts)
    meanCmaxoverdt = np.mean(Cmax/dt)
    
    # ---- Plot ----
    fig, axs = plt.subplots(4, 1, figsize=(4.5, 7.5), sharex=True)
   
    # 1) l2 norm
    axs[0].plot(dt, l2, marker='o')
    axs[0].axvline(1.4/meanCmaxoverdt, color='r', linestyle='--')
    axs[0].set_ylabel("$l_2$ norm")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].grid(True, which="both", ls="--", alpha=0.5)
    
    # 2) total scheme time
    axs[1].plot(dt, time_scheme, marker='o')
    axs[1].axvline(1.4/meanCmaxoverdt, color='r', linestyle='--')
    axs[1].set_ylabel("Total scheme\nwall-clock time (s)")
    axs[1].set_xscale("log")
    axs[1].grid(True, which="both", ls="--", alpha=0.5)
    
    # 3) time per step
    axs[2].plot(dt, time_per_step, marker='o')
    axs[2].axvline(1.4/meanCmaxoverdt, color='r', linestyle='--')
    axs[2].set_ylabel("Scheme wall-clock time\nper time step (s)")
    axs[2].set_yscale("log")
    axs[2].set_xscale("log")
    axs[2].grid(True, which="both", ls="--", alpha=0.5)
    
    ## 4) total iterations
    #axs[3].plot(dt, total_iters, marker='o')
    #axs[3].axvline(1.4/meanCmaxoverdt, color='r', linestyle='--')
    #axs[3].set_ylabel("Total iterations")
    #axs[3].set_xscale("log")
    #axs[3].grid(True, which="both", ls="--", alpha=0.5)    

    # 5) iterations per step
    axs[3].plot(dt[:-4], iterations_per_step[:-4], marker='o')
    axs[3].axvline(1.4/meanCmaxoverdt, color='r', linestyle='--')
    axs[3].set_ylabel("Iterations\nper time step")
    axs[3].set_xlabel("$\Delta t$")
    axs[0].secondary_xaxis('top', functions=(lambda x: meanCmaxoverdt*x, lambda x: meanCmaxoverdt*x)).set_xlabel("$C_{max}$")
    axs[3].set_yscale("log")
    axs[3].set_xscale("log")
    axs[3].grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    #figname = "timing_plot-20260622-itermin4-jiter4-nok3-log-thirdordermatrix"
    #figname = "timing_plot-20260616-itermin4-jiter4-nok3-log"
    figname = "timing_plot-20260709-noitermin_jiter4_IGn-log"
    plt.savefig(f"{figname}.pdf", dpi=300)
    plt.savefig(f"{figname}.svg", dpi=300)
    #plt.show()



def main():
    #filename = "output_timing_run_20x_fourthversion_20260615-itermin5-jiter5-k3.txt"#"singlerun_test_accuracy_timings-fourthversion-k3.txt" # "singlerun_test_accuracy_timings-fourthversion.txt" #"output_run_20x-fourthversion.txt"
    ###filename = "output_timing_run_20x_fourthversion_20260616-itermin4-jiter4-nok3.txt" # best one! (prior)
    filename = "output_timing_run_20x_fourthversion_20260709-noitermin_jiter4_IGn.txt"
    #filename = "output_timing_run_20x_fourthversion_20260622-itermin4-jiter4-nok3-thirdordermatrix.txt"
    #filename = "timing_test_20260614-itermin4.txt" # better (but also this one is the only average of two)
    #filename = "test_timing_20260615-itermin4-jiter4-nok3.txt"
    #filename = "test_timing_20260615-itermin3-jiter5-nok3.txt"
    #filename = "timing_test_20260614-itermin4-jiter4.txt"
    #filename = "timing_test_20260614-itermin5-jiter5.txt"
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