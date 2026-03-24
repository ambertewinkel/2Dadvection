#!/bin/bash

echo "First set of runs: unif nt500"
for i in {1..10}
do
    echo "Run #$i"
    py run_model.py config_swiftnondiv_sine-timing-unifnt500
    echo ""
done

echo "Second set of runs: ll nt500"
for i in {1..10}
do
    echo "Run #$i"
    py run_model.py config_swiftnondiv_sine-timing-llnt500
    echo ""
done

echo "Third set of runs: ll nt5000 dt0.02"
for i in {1..10}
do
    echo "Run #$i"
    py run_model.py config_swiftnondiv_sine-timing-llnt5000dt0_02
    echo ""
done