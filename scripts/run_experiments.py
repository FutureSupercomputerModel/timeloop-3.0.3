import subprocess

processes = []
for buffer_size in [2,20,100]:
    result_dir = f"result_fusion_{buffer_size}"
    subprocess.check_call(['mkdir', '-p', result_dir])
    stats_file = f"{result_dir}/stats.csv"
    output_file = f"{result_dir}/output.txt"
    with open(output_file, "w+") as f:
        print(['python3', 'sample-fusion.py', f'256-ws-{buffer_size}mb-sram.yaml', str(buffer_size), result_dir, stats_file])
        p = subprocess.Popen(['python3', 'sample-fusion.py', f'256-ws-{buffer_size}mb-sram.yaml', str(buffer_size), result_dir, stats_file], stdout=f)
        processes.append(p)

for p in processes:
    p.wait()
print('Done with buffer size: ', buffer_size)