#!/bin/bash

# Define the parameter arrays
activities=(0 8 16 32 48 64 72 80 88 96 104)
kappas=(32)
fictions=(10)
endn=(64)
randomloops=(1 2 3 4 5 6 7 8)
fracs=(0.1)
angmoms=(0.1 1 4 7 10 13 16 19)

mkdir -v position bondvector angleenergy force endtoend rg msd temp bondangle log rewritedata runfolder

# Output file to store commands
output_file="commands_to_run.txt"

# Clear the output file if it exists
> $output_file

# Loop over each combination of parameter values
for activity in "${activities[@]}"; do
    for kappa in "${kappas[@]}"; do
        for fiction in "${fictions[@]}"; do
            for end in "${endn[@]}"; do
                for randomloop in "${randomloops[@]}"; do
                    for frac in "${fracs[@]}"; do
                        for angmom in "${angmoms[@]}"; do

                            # Set the number of cores based on the frac value
                            if [ "$frac" = "0.05" ]; then
                                cores=2
                            elif [ "$frac" = "0.1" ]; then
                                cores=4
                            elif [ "$frac" = "0.2" ]; then
                                cores=6
                            elif [ "$frac" = "0.3" ]; then
                                cores=8
                            fi

                            # Generate random integers for the placeholders
                            randIntone=${RANDOM}
                            randInttwo=${RANDOM}
                            randIntthree=${RANDOM}
                            randIntfour=${RANDOM}

                            # Construct a unique output file name that includes every parameter
                            fileName="runfolder/in.activenoise.activity${activity}.kappa${kappa}.fiction${fiction}.endn${end}.rloop${randIntone}_${randInttwo}_${randIntthree}_${randIntfour}_.frac${frac}.angmom${angmom}.lmp"

                            # Update the input file based on the template file using sed.
                            # Adjust the regex patterns if needed to match the placeholders in your template.
                            sed -e "s/variable activity equal [0-9]*/variable activity equal $activity/" \
                                -e "s/variable kappa equal [0-9]*/variable kappa equal $kappa/" \
                                -e "s/variable fiction equal [0-9.]*/variable fiction equal $fiction/" \
                                -e "s/variable endn equal [0-9]*/variable endn equal $end/" \
                                -e "s/variable frac equal [0-9.]*/variable frac equal $frac/" \
                                -e "s/variable angmom equal [0-9.]*/variable angmom equal $angmom/" \
                                -e "s/variable randIntone equal [0-9]*/variable randIntone equal $randIntone/" \
                                -e "s/variable randInttwo equal [0-9]*/variable randInttwo equal $randInttwo/" \
                                -e "s/variable randIntthree equal [0-9]*/variable randIntthree equal $randIntthree/" \
                                -e "s/variable randIntfour equal [0-9]*/variable randIntfour equal $randIntfour/" \
                                in.explicitactivebathpolymer.lmp > "$fileName"

                            # Append the mpirun command with the correct number of cores and file name to the output file
                            echo "mpirun -np $cores ~/lammps/src/lmp_mpi -in $fileName" >> $output_file

                        done
                    done
                done
            done
        done
    done
done

echo "All commands saved to $output_file"
