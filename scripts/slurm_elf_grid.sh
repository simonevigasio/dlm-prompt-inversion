#!/usr/bin/env bash
# Run the ELF experiment grid on AAU AI Cloud: one Slurm array task per GPU, each task one
# shard of scripts/run_elf_grid.py. Submit from the repository root:
#
#     sbatch scripts/slurm_elf_grid.sh
#     squeue --me                                  # what is queued or running
#     tail -f elf-grid_<job>_<task>.out            # follow one worker
#     python3 scripts/run_elf_grid.py --list       # state of every experiment
#
# The number of workers is the size of the array below; change only that line. A task that
# hits the time limit (or fails) loses nothing: submit the script again and finished
# experiments are skipped, unfinished ones resume.

#SBATCH --job-name=elf-grid
#SBATCH --partition=prioritized
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=elf-grid_%A_%a.out
#SBATCH --error=elf-grid_%A_%a.err

# Container whose python3 runs the experiment. It must be the one the repository's .venv was
# created with (run_pipeline.py switches to .venv by itself). Empty: no container, plain python3.
container_image="${CONTAINER_IMAGE-/home/container/pytorch/pytorch_25.04.sif}"

cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the repository root}" || exit 1
command=(python3 scripts/run_elf_grid.py --device gpu
         --shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}")
echo "$(hostname): shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}, container: ${container_image:-none}"

if [ -n "$container_image" ]; then
    singularity exec --nv "$container_image" "${command[@]}"
else
    "${command[@]}"
fi
