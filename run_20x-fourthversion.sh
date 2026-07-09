#!/bin/bash

echo "Started script"
sleep 14400
echo "Started"

for i in {1..20}
do
    echo "Runs #$i"
    echo ""
    echo "-------------- nt 50 dt 2.0 ll --------------"
    py run_model.py config_swiftnondiv_sine-timing-llnt50dt2_0   
    echo ""
    echo "-------------- nt 80 dt 1.25 ll --------------"
    py run_model.py config_swiftnondiv_sine-timing-llnt80dt1_25
    echo ""
    echo "-------------- nt 100 dt 1.0 ll --------------"
    py run_model.py config_swiftnondiv_sine-timing-llnt100dt1_0   
    echo ""
    echo "-------------- nt 125 dt 0.8 ll --------------"
    py run_model.py config_swiftnondiv_sine-timing-llnt125dt0_8
    echo ""    
    echo "-------------- nt 160 dt 0.625 ll --------------"
    py run_model.py config_swiftnondiv_sine-timing-llnt160dt0_625
    echo ""
    echo "-------------- nt 200 dt 0.5 ll --------------"
    py run_model.py config_swiftnondiv_sine-timing-llnt200dt0_5
    echo ""
    echo "-------------- nt 250 dt 0.4 ll --------------"
    py run_model.py config_swiftnondiv_sine-timing-llnt250dt0_4
    echo ""
    echo "-------------- nt 320 dt 0.3125 ll --------------"
    py run_model.py config_swiftnondiv_sine-timing-llnt320dt0_3125
    echo ""
    echo "-------------- nt 400 dt 0.25 ll --------------"
    py run_model.py config_swiftnondiv_sine-timing-llnt400dt0_25
    echo ""
    echo "-------------- nt 500 dt 0.2 ll --------------"
    py run_model.py config_swiftnondiv_sine-timing-llnt500dt0_2
    echo ""
    echo "-------------- nt 625 dt 0.16 ll --------------"
    py run_model.py config_swiftnondiv_sine-timing-llnt625dt0_16
    echo ""
    echo "-------------- nt 800 dt 0.125 ll --------------"
    py run_model.py config_swiftnondiv_sine-timing-llnt800dt0_125
    echo ""
    echo "-------------- nt 1000 dt 0.1 ll --------------"
    py run_model.py config_swiftnondiv_sine-timing-llnt1000dt0_1
    echo ""
    echo "-------------- nt 1250 dt 0.08 ll --------------"
    py run_model.py config_swiftnondiv_sine-timing-llnt1250dt0_08
    echo ""
    echo "-------------- nt 1600 dt 0.0625 ll --------------"
    py run_model.py config_swiftnondiv_sine-timing-llnt1600dt0_0625
    echo ""
    echo "-------------- nt 2000 dt 0.05 ll --------------"
    py run_model.py config_swiftnondiv_sine-timing-llnt2000dt0_05
    echo ""
    echo "-------------- nt 2500 dt 0.04 ll --------------"
    py run_model.py config_swiftnondiv_sine-timing-llnt2500dt0_04
    echo ""
    echo "-------------- nt 3200 dt 0.03125 ll --------------"
    py run_model.py config_swiftnondiv_sine-timing-llnt3200dt0_03125
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

echo "Done"
