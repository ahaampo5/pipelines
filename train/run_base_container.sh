NUM_GPUS=all
WORKSPACE=/mnt/data01/workspace/jckim:/root/workspace
HUGGINGFACE_CACHE=/mnt/data01/huggingface/cache
CONTAINER_NAME=server_jckim
SHARED_MEMORY=4G
DOCKER_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

docker run -dit \
	--net host \
	--gpus $NUM_GPUS \
	-v $WORKSPACE:/root/workspace \
	-v $HUGGINGFACE_CACHE:/root/.cache/huggingface \
	--name $CONTAINER_NAME \
	--shm-size $SHARED_MEMORY \
	$DOCKER_IMAGE

