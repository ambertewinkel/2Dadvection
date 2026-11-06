#!/bin/bash

# Runs the four BK25 testcases of the bk25 figure in plot_paper_results.py and
# animates them over one full revolution with plot_paper_bk25_animation.py.
# Every time level is stored in ./output_paper/<outputdir>/data/. An existing
# <outputdir> is not overwritten: the run then goes to <outputdir>_<i>, while
# the animation (like plot_paper_results.py) reads <outputdir>.

RUN="python run_model.py"

for cfg in config_swiftnondiv_slotcyl_unif \
           config_swiftnondiv_slotcyl_unif_FCT \
           config_swiftnondiv_slotcyl_llgp \
           config_swiftnondiv_slotcyl_llgp_FCT; do
    echo "=== $cfg"
    $RUN "$cfg"
done

python plot_paper_bk25_animation.py

echo "Done"
