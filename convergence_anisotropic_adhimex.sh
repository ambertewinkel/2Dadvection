#!/usr/bin/env bash

# Runs run_model.py and then plot_tracer.py once per entry in INPUTS


INPUTS=(
    #"constantuv_sine_unif_anis_ny1280dt0_2"
    #"constantuv_sine_unif_ny1280dt0_2"
    #"constantuv_sine_unif_anis_ny640dt0_4"
    #"constantuv_sine_unif_ny640dt0_4"    
    #"constantuv_sine_unif_anis_ny2560dt0_1"
    #"constantuv_sine_unif_ny2560dt0_1"
    #"swiftnondiv_sine_unif_anis_ny1280dt0_2"
    #"swiftnondiv_sine_unif_ny1280dt0_2"
    #"swiftnondiv_sine_unif_anis_ny640dt0_4"
    #"swiftnondiv_sine_unif_ny640dt0_4"    
    #"swiftnondiv_sine_unif_anis_ny2560dt0_1"
    #"swiftnondiv_sine_unif_ny2560dt0_1"
    #"constantv_sine_unif_anis_ny640dt0_4"
    #"constantv_sine_unif_ny640dt0_4"    
    #"constantv_sine_unif_anis_ny1280dt0_2"
    #"constantv_sine_unif_ny1280dt0_2"
    #"constantv_sine_unif_anis_ny2560dt0_1"
    #"constantv_sine_unif_ny2560dt0_1"
    #"constantu_sine_unif_anis_ny640dt0_4"
    #"constantu_sine_unif_ny640dt0_4"    
    #"constantu_sine_unif_anis_ny1280dt0_2"
    #"constantu_sine_unif_ny1280dt0_2"
    #"constantu_sine_unif_anis_ny2560dt0_1"
    #"constantu_sine_unif_ny2560dt0_1"
    # First set of runs 17-09-2026:
    ###"constantuv_sine_unif_anis_ny1280dt0_2nt2"
    ###"constantuv_sine_unif_ny1280dt0_2nt2"
    ###"constantuv_sine_unif_anis_ny640dt0_4nt1"
    ###"constantuv_sine_unif_ny640dt0_4nt1"
    ###"constantuv_sine_unif_anis_ny2560dt0_1nt4"
    ###"constantuv_sine_unif_ny2560dt0_1nt4"
    ###"constantv_sine_unif_anis_ny640dt0_4nt1"
    ###"constantv_sine_unif_ny640dt0_4nt1"
    ###"constantv_sine_unif_anis_ny1280dt0_2nt2"
    ###"constantv_sine_unif_ny1280dt0_2nt2"
    ###"constantv_sine_unif_anis_ny2560dt0_1nt4"
    ###"constantv_sine_unif_ny2560dt0_1nt4"
    ###"constantu_sine_unif_anis_ny640dt0_4nt1"
    ###"constantu_sine_unif_ny640dt0_4nt1" 
    ###"constantu_sine_unif_anis_ny1280dt0_2nt2"
    ###"constantu_sine_unif_ny1280dt0_2nt2"
    ###"constantu_sine_unif_anis_ny2560dt0_1nt4"
    ###"constantu_sine_unif_ny2560dt0_1nt4"
    # Second set of runs 17-09-2026
    ###"constantuvsmallu_sine_unif_anis_ny1280dt0_2nt2"
    ###"constantuvsmallu_sine_unif_ny1280dt0_2nt2"
    ###"constantuvsmallu_sine_unif_anis_ny640dt0_4nt1"
    ###"constantuvsmallu_sine_unif_ny640dt0_4nt1"
    ###"constantuvsmallu_sine_unif_anis_ny2560dt0_1nt4"
    ###"constantuvsmallu_sine_unif_ny2560dt0_1nt4"
    # Third set of runs 17-09-2026
    "constantuvlargev_sine_unif_anis_ny128dt0_2nt2"
    "constantuvlargev_sine_unif_ny128dt0_2nt2"
    "constantuvlargev_sine_unif_anis_ny64dt0_4nt1"
    "constantuvlargev_sine_unif_ny64dt0_4nt1"
    "constantuvlargev_sine_unif_anis_ny256dt0_1nt4"
    "constantuvlargev_sine_unif_ny256dt0_1nt4"
)

for args in "${INPUTS[@]}"; do
    echo "=== Running: python run_model.py config_$args ==="
    python run_model.py config_$args
    python plot_tracer.py swift dated/20260917/$args
    #python error.py dated/20260917/$args finaltoinitial
    python error.py dated/20260917/$args finaltoanalytic
done