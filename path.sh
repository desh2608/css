# Print immediately
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PYTHONPATH}:`pwd`/scripts/python

# Activate environment
. /home/draj/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate ovad

export LC_ALL=C
