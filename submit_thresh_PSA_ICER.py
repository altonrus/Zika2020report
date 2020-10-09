import os, sys, time

nprocs = 200

for rank in range(nprocs):
    with open("Threshold_PSA_ICER_%03d.sbatch" % (rank), 'w') as f:
      f.write("#!/bin/bash\n")
      f.write("\n")
      f.write("#SBATCH --job-name=Threshold_PSA_ICER_%03d\n" % (rank))
      f.write("#SBATCH --time=30:00:00\n")
      f.write("#SBATCH --ntasks=1\n")
      f.write("#SBATCH --cpus-per-task=1\n")
      f.write("#SBATCH --mem-per-cpu=2G\n")
      f.write("#SBATCH --nodes=1\n")
      f.write("#SBATCH --ntasks-per-node=1\n")
      if rank == nprocs-1:
          f.write("#SBATCH --mail-type=END\n")
          f.write("#SBATCH --mail-user=altonr@stanford.edu\n")
      f.write("\n")
      f.write("module load py-scipystack\n")
      f.write("python ZIKV_sim_threshold.py threshold_PSA_ICER_%03d 50 1"% (rank))
    os.system("sbatch Threshold_PSA_ICER_%03d.sbatch" % (rank))
    time.sleep(1)

