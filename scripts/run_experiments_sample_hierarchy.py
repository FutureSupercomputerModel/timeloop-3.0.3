import subprocess

processes = []
for buffer_size in [2]:
    result_dir = f"result_fusion_sample_hierarchy_{buffer_size}"
    subprocess.check_call(['mkdir', '-p', result_dir])
    # stats_file = f"{result_dir}/stats.csv"
    output_file = f"{result_dir}/output.txt"
    with open(output_file, "w+") as f:
        print(['python3', 'sample-fusion.py', f'sample-hierarchy-no-dram-{buffer_size}MB-sram.yaml', f'sample-hierarchy.yaml', str(buffer_size), result_dir])
        p = subprocess.run(['python3', 'sample-fusion.py', f'sample-hierarchy-no-dram-{buffer_size}MB-sram.yaml', f'sample-hierarchy.yaml', str(buffer_size), result_dir], stdout=f)
        # processes.append(p)

# for p in processes:
#     p.wait()
print('Done')