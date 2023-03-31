NETWORK_NAME=Caricature
NETWORK_PKL=Caricature.pkl
NETWORK_DIR="./pretrained/$(printf '%s' $NETWORK_NAME)"
OUTDIR="./logs/gen_samples/$(printf '%s' $NETWORK_NAME)"
CUDA_VISIBLE_DEVICES=0 python gen_samples.py --network_dir $NETWORK_DIR --seeds 0-2 --trunc 0.7 --outdir $OUTDIR --network_pkl $NETWORK_PKL --rot_angle 0.785 --rot_angle_gap 0.261 --shapes True --reload_modules True