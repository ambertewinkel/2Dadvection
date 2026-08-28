#!/bin/bash

RUN="python run_model.py"
REPEATS=20
TIMING_LOG="output_paper/output_timing_run_${REPEATS}x.txt"

for cfg in config_constancy_cp \
           config_constancy_ncp \
           config_hadley_adhimex \
           config_hadley_ex \
           config_swiftnondiv_slotcyl_llgp \
           config_swiftnondiv_slotcyl_llgp_FCT \
           config_swiftnondiv_slotcyl_unif \
           config_swiftnondiv_slotcyl_unif_FCT; do
    echo "=== $cfg"
    $RUN "$cfg"
done

python plot_paper_results.py
python graphical_abstract.py

# Timing configs
TIMING_CONFIGS=(
    config_swiftnondiv_sine-timing-llnt50dt2_0
    config_swiftnondiv_sine-timing-llnt80dt1_25
    config_swiftnondiv_sine-timing-llnt100dt1_0
    config_swiftnondiv_sine-timing-llnt125dt0_8
    config_swiftnondiv_sine-timing-llnt160dt0_625
    config_swiftnondiv_sine-timing-llnt200dt0_5
    config_swiftnondiv_sine-timing-llnt250dt0_4
    config_swiftnondiv_sine-timing-llnt320dt0_3125
    config_swiftnondiv_sine-timing-llnt400dt0_25
    config_swiftnondiv_sine-timing-llnt500dt0_2
    config_swiftnondiv_sine-timing-llnt625dt0_16
    config_swiftnondiv_sine-timing-llnt800dt0_125
    config_swiftnondiv_sine-timing-llnt1000dt0_1
    config_swiftnondiv_sine-timing-llnt1250dt0_08
    config_swiftnondiv_sine-timing-llnt1600dt0_0625
    config_swiftnondiv_sine-timing-llnt2000dt0_05
    config_swiftnondiv_sine-timing-llnt2500dt0_04
    config_swiftnondiv_sine-timing-llnt3200dt0_03125
    config_swiftnondiv_sine-timing-llnt4000dt0_025
    config_swiftnondiv_sine-timing-llnt5000dt0_02
)

for ((i = 1; i <= REPEATS; i++)); do
    echo "Runs #$i"
    echo ""
    for cfg in "${TIMING_CONFIGS[@]}"; do
        # config_swiftnondiv_sine-timing-llnt50dt2_0 -> nt 50, dt 2.0
        label=${cfg#config_swiftnondiv_sine-timing-ll}
        nt=${label%%dt*}
        nt=${nt#nt}
        dt=${label#*dt}
        dt=${dt/_/.}

        echo "-------------- nt $nt dt $dt ll --------------"
        $RUN "$cfg"
        echo ""
    done
    echo ""
    echo ""
done > "$TIMING_LOG" 2>&1

python plot_paper_timing.py

echo "Done"