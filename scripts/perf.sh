PROG_PATH="/home/yama/mfs/ZhuSuan-intra-node/examples"
LOG_PATH="/home/yama/mfs/logs"

PROG="vae_conv"
PROG="vae"

NOW=`date +%Y%m%d%H%M%S`

#declare -a BATCH_SIZE=(100 200 400)
declare -a BATCH_SIZE=(100)
declare -a LB_SAMPLES=(1 8 64)
declare -a NUM_GPUS=(1 2 4)

MASTER_DEVICE="/gpu:0"

for bs in "${BATCH_SIZE[@]}"
do
	for lb in "${LB_SAMPLES[@]}"
	do
		for ng in "${NUM_GPUS[@]}"
		do
			CMD="python $PROG_PATH/$PROG.py
				--batch_size=$bs
				--lb_samples=$lb
				--num_gpus=$ng
				--master_device=$MASTER_DEVICE"
			echo $CMD
			$CMD 2>&1 | tee $LOG_PATH/$PROG-[$bs]bsize-[$lb]lbs-[$ng]gpus-$NOW.log
			sleep 10
		done
	done
done

