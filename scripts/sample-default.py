#! /usr/bin/env python3

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import inspect
import os
import pprint
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml

this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
this_directory = os.path.dirname(this_file_path)
root_dir = os.path.join(os.path.dirname(this_file_path), '..')

sys.path.append(os.path.join(root_dir, 'scripts'))
from cnn_layers import *
import timeloop
import parse_timeloop_output
import layerFuser

if len(sys.argv) > 3:
    config_file     = sys.argv[1]
    raw_result_dir  = sys.argv[2]
    stats_dir       = sys.argv[3]
else:
    print("Usage: config.yaml run/ results.csv")
    sys.exit(1)

#config_abspath = os.path.join(root_dir, 'configs/mapper/sample-hierarchy.yaml')
#config_abspath = os.path.join(root_dir, 'configs/mapper/chen-asplos2014.yaml')
config_abspath = os.path.join(root_dir, 'configs/mapper/' + str(config_file))

# Create array to store important stats  
cycles_list = [] #cycles
energy_list = [] #energy_pJ
energy_per_mac_list = [] #energy_per_mac
macs_num_list = [] #macs

# Create total stats variables
total_cycles = 0 
total_energy_net = 0 

# Just test that path points to a valid config file.
with open(config_abspath, 'r') as f:
    config = yaml.full_load(f)


for i in range(0, len(cnn_layers)):
    
    problem = cnn_layers[i]

    print("Preparing to run timeloop for problem index ", i)

    dirname = str(raw_result_dir) + 'problem_' + str(i) + '/'
    subprocess.check_call(['mkdir', '-p', dirname])

    timeloop.run_timeloop(dirname, configfile = config_abspath, workload_bounds = problem)

    stats = parse_timeloop_output.parse_timeloop_stats(dirname)
    if stats == {}:
        print("Timeloop couldn't find a mapping for this problem within the search parameters, please check the log for more details.")
    else:
        print("Run successful, see log for text stats, or use the Python parser to parse the XML stats.")
        print("Stats from run:")
        pprint.pprint(stats)
        total_cycles+=stats['cycles']
        total_energy_net+=stats['energy_pJ']
        print("problem cycles: ", stats['cycles'])
        print("problem total energy (pJ): ", stats['energy_pJ'])
        cycles_list.append(stats['cycles'])
        energy_list.append(stats['energy_pJ'])
        energy_per_mac_list.append(stats['energy_per_mac'])
        macs_num_list.append(stats['macs'])

        cycles_array = np.array(cycles_list)
        energy_array = np.array(energy_list)
        energy_per_mac_array = np.array(energy_per_mac_list)
        macs_num_array = np.array(macs_num_list)

result_stats = np.column_stack((np.arange(len(cnn_layers)), cycles_array, energy_array, energy_per_mac_array, macs_num_array))

np.savetxt(stats_dir, result_stats, delimiter=',', header='i, cycles, energy, energy per mac, macs', comments='')

print("DONE.")
print("Total cycles: ", total_cycles)
