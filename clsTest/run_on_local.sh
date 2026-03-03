#!/bin/bash

ulimit -s unlimited

#mpirun --mca osc ucx -np 1 nekrs --cimode 1 --setup tls 2>&1 | tee "log.run"
mpirun --mca osc ucx -np 1 nekrs --setup tls 2>&1 | tee "log.run"
