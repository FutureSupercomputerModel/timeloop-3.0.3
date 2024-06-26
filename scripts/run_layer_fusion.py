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
import layerFuserRecursive
import layerFuserRecursiveDP
import layerFuserHelper as helper
import matplotlib.pyplot as plt


colors = ["red", "blue", "green", "orange"]
shapes = ["o", "s", "D", "^"]
for buffer_size in [1, 2, 20, 100]:
    strategies = layerFuserRecursiveDP.fuse_layer_recursive_start("", cnn_layers, pooling_layers, buffer_size)
    helper.printStrategies(strategies)
    macs = []
    offchip_access = []
    for i in range(len(strategies)):
        total_macs = helper.get_total_macs(strategies[i])
        total_offchip_access = helper.get_total_offchip_access(strategies[i])
        macs.append(total_macs)
        offchip_access.append(total_offchip_access)
    
    plt.scatter(macs, offchip_access, color = colors.pop(0), alpha = 0.3, marker = shapes.pop(0), label = f"Buffer Size: {buffer_size}MB")
plt.xlabel("Total MACs")
plt.ylabel("Total Off-chip Access")
plt.legend()
plt.show()
plt.savefig(f"macs_vs_offchip_access.png",dpi=300)
# helper.summarizeStrategies(strategies, buffer_size)

# strategy_index=0
# for fused_groups in strategies:

#     index = 0
#     cycles_list = [] #cycles
#     energy_list = [] #energy_pJ
#     energy_per_mac_list = [] #energy_per_mac
#     macs_num_list = [] #macs
#     for i in range(0, len(fused_groups)):
#         for j in range(0, len(fused_groups[i])):
            
#             # print(fused_groups[i][j])
#             input_tile_count = fused_groups[i][j][3]
#             print("input_tile_count: ", input_tile_count)
#             fused_groups[i][j][3]=1
#             problem = fused_groups[i][j]

#             print("Preparing to run timeloop for problem index ", index)
#             print("Problem: ", problem)

#             dirname = str(raw_result_dir) + '/strategy_' + str(strategy_index)+ '/problem_' + str(index) + '/'
#             subprocess.check_call(['mkdir', '-p', dirname])
#             if len(fused_groups[i])>1:
#                 timeloop.run_timeloop(dirname, configfile = config_no_dram_abspath, workload_bounds = problem)
#             else:
#                 timeloop.run_timeloop(dirname, configfile = config_with_dram_abspath, workload_bounds = problem)

#             stats = parse_timeloop_output.parse_timeloop_stats(dirname)
#             if stats == {}:
#                 print("Timeloop couldn't find a mapping for this problem within the search parameters, please check the log for more details.")
#             else:
#                 print("Run successful, see log for text stats, or use the Python parser to parse the XML stats.")
#                 print("Stats from run:")
#                 pprint.pprint(stats)
#                 # cycles
#                 cycles_all_tiles = stats['cycles'] * input_tile_count
#                 total_cycles+=cycles_all_tiles
#                 print("problem cycles: ", cycles_all_tiles)
#                 # energy
#                 energy_all_tiles = stats['energy_pJ'] * input_tile_count
#                 total_energy_net+=energy_all_tiles
#                 print("problem total energy (pJ): ", energy_all_tiles)
#                 # macs
#                 macs_all_tiles = stats['macs'] * input_tile_count
#                 #append lists
#                 cycles_list.append(cycles_all_tiles)
#                 energy_list.append(energy_all_tiles)
#                 energy_per_mac_list.append(stats['energy_per_mac'])
#                 macs_num_list.append(macs_all_tiles)

#             index+=1

#     cycles_array = np.array(cycles_list)
#     energy_array = np.array(energy_list)
#     energy_per_mac_array = np.array(energy_per_mac_list)
#     macs_num_array = np.array(macs_num_list)

#     result_stats = np.column_stack((np.arange(index), cycles_array, energy_array, energy_per_mac_array, macs_num_array))
#     stats_dir = f"{raw_result_dir}/strategy_{strategy_index}/stats.csv"
#     np.savetxt(stats_dir, result_stats, delimiter=',', header='i, cycles, energy, energy per mac, macs', comments='')

#     print("DONE.")
#     print("Total cycles: ", total_cycles)

#     strategy_index += 1
