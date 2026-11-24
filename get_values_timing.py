import re
import argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def parse_file(filename, aniso=False, scipy=None):
    # scipy: None keeps every run whatever its linear solver (the behaviour without
    # --gmresmscipy); True keeps only the SciPy GMRES(m) runs, False only the other ones.
    data = defaultdict(lambda: defaultdict(list))

    current_nt = None
    current_is_aniso = None  # which sub-run (sub-block) the following lines belong to
    current_is_scipy = None  # whether that sub-run used the SciPy GMRES(m)

    # Strict nt pattern: must start with dashes and end with dashes
    nt_pattern = re.compile(r"^-{14}\s*nt\s+(\d+)")
    # Each nt block contains an iso sub-run and, if present, an aniso sub-run,
    # one after the other. This line marks the start of each one.
    outputfile_pattern = re.compile(r"See output file\s+(\S+)")

    patterns = {
        "time_total": re.compile(r"Time total time stepping:\s*([0-9eE\.\-\+]+)"),
        "time_vel": re.compile(r"Time total velocity:\s*([0-9eE\.\-\+]+)"),
        "time_scheme": re.compile(r"Time total scheme:\s*([0-9eE\.\-\+]+)"),
        "total_iters": re.compile(r"Total number of iterations over all times:\s*([0-9eE\.\-\+]+)"),
        "l2norm": re.compile(r"l2norm\s*=\s*([0-9eE\.\-\+]+)"),
        "linfnorm": re.compile(r"linfnorm\s*=\s*([0-9eE\.\-\+]+)"),
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
                current_is_aniso = None
                current_is_scipy = None
                continue

            if current_nt is None:
                continue

            # Figure out which sub-run we just entered. Recognise both the
            # '_anis_' and '_aniso_' filename spellings as the aniso sub-run, and
            # a 'gmresmscipy' tag in the filename as a SciPy GMRES(m) run.
            outputfile_match = outputfile_pattern.search(line)
            if outputfile_match:
                current_is_aniso = "_anis_" in outputfile_match.group(1) or "_aniso_" in outputfile_match.group(1)
                current_is_scipy = "gmresmscipy" in outputfile_match.group(1)
                continue

            # Only keep lines belonging to the sub-run we were asked for
            if current_is_aniso != aniso:
                continue
            if scipy is not None and current_is_scipy != scipy:
                continue

            # Scalar variables
            for key in ["time_total", "time_vel", "time_scheme", "total_iters", "l2norm", "linfnorm"]:
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


def plot_timings(data_list):
    # data_list: list of (data, label, color, marker) tuples, one entry per line to plot.
    # A single entry reproduces the original single-line plot exactly.

    # ---- Plot ----
    fig, axs = plt.subplots(5, 1, figsize=(4.5, 8.5), sharex=True)
    meanCmaxoverdt_list = []

    for data, label, color, marker in data_list:
        nts = sorted(data.keys())

        # Extract data
        dt = np.array([data[nt]["dt"] for nt in nts])
        Cmax  = np.array([data[nt]["C_max"]  for nt in nts])

        l2 = np.array([data[nt]["l2norm"] for nt in nts])
        linf = np.array([data[nt]["linfnorm"] for nt in nts])
        time_scheme = np.array([np.mean(data[nt]["time_scheme"]) for nt in nts])
        total_iters = np.array([data[nt]["total_iters"] for nt in nts])

        time_per_step = time_scheme / np.array(nts)
        iterations_per_step = total_iters / np.array(nts)
        meanCmaxoverdt = np.mean(Cmax/dt)
        meanCmaxoverdt_list.append(meanCmaxoverdt)

        # 1) l2 norm
        axs[0].plot(dt, l2, marker=marker, color=color, label=label)
        axs[0].axvline(1.4/meanCmaxoverdt, color='r', linestyle='--')

        # 2) linf norm
        axs[1].plot(dt, linf, marker=marker, color=color, label=label)
        axs[1].axvline(1.4/meanCmaxoverdt, color='r', linestyle='--')

        # 3) total scheme time
        axs[2].plot(dt, time_scheme, marker=marker, color=color, label=label)
        axs[2].axvline(1.4/meanCmaxoverdt, color='r', linestyle='--')

        # 4) time per step
        axs[3].plot(dt, time_per_step, marker=marker, color=color, label=label)
        axs[3].axvline(1.4/meanCmaxoverdt, color='r', linestyle='--')

        # 5) iterations per step
        axs[4].plot(dt[:-4], iterations_per_step[:-4], marker=marker, color=color, label=label)
        axs[4].axvline(1.4/meanCmaxoverdt, color='r', linestyle='--')

    axs[0].set_ylabel("$l_2$ norm")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].grid(True, which="both", ls="--", alpha=0.5)
    meanCmaxoverdt = np.mean(meanCmaxoverdt_list)
    axs[0].secondary_xaxis('top', functions=(lambda x: meanCmaxoverdt*x, lambda x: meanCmaxoverdt*x)).set_xlabel("$C_{max}$")
    
    axs[1].set_ylabel("$l_\\infty$ norm")
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].grid(True, which="both", ls="--", alpha=0.5)

    axs[2].set_ylabel("Total scheme\nwall-clock time (s)")
    axs[2].set_xscale("log")
    axs[2].grid(True, which="both", ls="--", alpha=0.5)

    axs[3].set_ylabel("Scheme wall-clock time\nper time step (s)")
    axs[3].set_yscale("log")
    axs[3].set_xscale("log")
    axs[3].grid(True, which="both", ls="--", alpha=0.5)

    axs[4].set_ylabel("Iterations\nper time step")
    axs[4].set_xlabel("$\\Delta t$")
    axs[4].set_yscale("log")
    axs[4].set_xscale("log")
    axs[4].grid(True, which="both", ls="--", alpha=0.5)

    if len(data_list) > 1:
        axs[0].legend()

    plt.tight_layout()
    #figname = "timing_plot-20260916and17-anis"
    #figname = "timing_plot_adhimex_anis_adjusted_20260923-removedlastbit-and16002000"
    #figname = "timing_noiters_accuracy_20260930"
    #figname = "output_1x_linfnormtest_20261106"
    #figname = "output_1x_gmresmscipytest_20261106"
    #figname = "output_comparing_gmresm-to-gmresmscipy_20261107"
    #figname = "output_adhimex_tol1em7_1x_20261123"
    figname = "output_adlimhexrk4_20261124"

    plt.savefig(f"{figname}.pdf", dpi=300)
    plt.savefig(f"{figname}.svg", dpi=300)
    #plt.show()


def load_dataset(filename, aniso, scipy=None):
    data = parse_file(filename, aniso=aniso, scipy=scipy)
    if not data:
        return None
    data = turn_into_np_arrays(data)
    data = add_mean_timings(data)

    simulated_time = 100.0
    for nt, values in data.items():
        data[nt]["dt"] = simulated_time / nt

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bothanisoandiso", action="store_true",
                         help="Plot the aniso runs alongside the iso ones.")
    parser.add_argument("--gmresmscipy", action="store_true",
                         help="Plot the GMRES(m) runs (x markers) and the SciPy GMRES(m) runs "
                              "(+ markers; 'gmresmscipy' in their output file name) as separate curves.")
    args = parser.parse_args()

    #filename = "timing_output_anis_20260916and17.txt"
    #filename = "timing_output_adhimex_anis_adjusted_20260923-removedlastbit-and16002000.txt"
    #filename = "timing_noiters_accuracy_20260930.txt"
    #filename = "output_1x_linfnormtest_20261106.txt"
    #filename = "output_1x_gmresmscipytest_20261106.txt"
    #filename = "output_comparing_gmresm-to-gmresmscipy_20261107.txt"
    #filename = "output_adhimex_tol1em7_1x_20261123.txt"
    filename = "output_adlimhexrk4_20261124.txt"

    # Sub-runs to plot: (aniso, label, color, color of its SciPy GMRES(m) curve)
    groups = [(False, "iso", "blue", "darkorange")]
    if args.bothanisoandiso:
        groups.append((True, "aniso", "green", "mediumvioletred"))

    # Linear solvers to split each sub-run into, one marker each: (scipy, label, marker).
    # Without --gmresmscipy, scipy=None keeps all runs together in one curve, as before.
    if args.gmresmscipy:
        solvers = [(False, "GMRES(m)", "x"), (True, "SciPy GMRES(m)", "+")]
    else:
        solvers = [(None, None, "x")]

    data_list = []
    for aniso, group_label, color, scipy_color in groups:
        for scipy, solver_label, marker in solvers:
            if solver_label is None:                  # no --gmresmscipy: the old labels, "iso" / "aniso"
                label = group_label
            elif len(groups) == 1:                    # --gmresmscipy, iso only: "iso" adds nothing, (the order matters: elif needs to come after if here)
                label = solver_label                  #   so just "GMRES(m)" / "SciPy GMRES(m)"
            else:                                     # both flags: 4 curves, so both names are needed
                label = f"{group_label} {solver_label}"

            data = load_dataset(filename, aniso=aniso, scipy=scipy)
            if data is None:
                print(f"Warning: no {label} runs found in the file; not plotting them.")
                continue

            for nt, values in data.items():
                print(f"[{label}] nt: {nt}, dt: {values['dt']:.2e}, C_min: {values['C_min']:.2e}, C_max: {values['C_max']:.2e}, l2norm: {values['l2norm']:.2e}, linfnorm: {values['linfnorm']:.2e}, time_scheme_mean: {values['time_scheme_mean']:.2f}s, total_iters: {values['total_iters']}")
            data_list.append((data, label, scipy_color if scipy else color, marker))

    plot_timings(data_list)

if __name__ == "__main__":
    main()