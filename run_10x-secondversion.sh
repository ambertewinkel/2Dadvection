#!/bin/bash

for i in {1..10}
do
    echo "Runs #$i"
    echo ""
    echo "-------------- nt 500 dt 0.2 unif--------------"
    py run_model.py config_swiftnondiv_sine-timing-unifnt500
    echo ""
    echo "-------------- nt 500 dt 0.2 ll --------------"
    py run_model.py config_swiftnondiv_sine-timing-llnt500
    echo ""
    echo "-------------- nt 1000 dt 0.1 ll --------------"
    py run_model.py config_swiftnondiv_sine-timing-llnt1000dt0_1
    echo ""
    echo "-------------- nt 2000 dt 0.05 ll --------------"
    py run_model.py config_swiftnondiv_sine-timing-llnt2000dt0_05
    echo ""
    echo "-------------- nt 2500 dt 0.04 ll --------------"
    py run_model.py config_swiftnondiv_sine-timing-llnt2500dt0_04
    echo ""
    echo "-------------- nt 4000 dt 0.025 ll --------------"
    py run_model.py config_swiftnondiv_sine-timing-llnt4000dt0_025
    echo ""
    echo "-------------- nt 5000 dt 0.02 ll --------------"
    py run_model.py config_swiftnondiv_sine-timing-llnt5000dt0_02
    echo ""
    echo ""
    echo ""
done