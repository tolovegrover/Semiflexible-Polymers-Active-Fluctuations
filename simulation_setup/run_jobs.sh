#!/bin/bash

# Set cores available per node
cores_per_node=32

# Clean up any previous PBS job scripts that may be present
rm -f sim_part_*.pbs

job_count=0
# Loop over each pre-packed job file. (They were created as job_0.txt, job_1.txt, etc.)
for job_file in job_*.txt; do
    job_count=$((job_count + 1))
    job_name="${job_count}.RW_lj"

    # Create a PBS script for this job file
    cat > ${job_name}.pbs << EOF
#!/bin/bash
#PBS -N ${job_name}
#PBS -l nodes=1:ppn=${cores_per_node}
#PBS -q default
#PBS -o ${job_name}.out
#PBS -e ${job_name}.err
#PBS -V

# Change to the working directory
cd \$PBS_O_WORKDIR

# Load GNU parallel if required
module load parallel

# Run the job commands in parallel on ${cores_per_node} cores.
parallel -j${cores_per_node} :::: ${job_file}
EOF

    # Submit the job to PBS
    qsub ${job_name}.pbs
done

echo "Submitted ${job_count} jobs."
