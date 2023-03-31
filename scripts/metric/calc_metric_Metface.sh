METRIC=fid50k_full

PKL_RANGE=1,2017
NETWORK_DIR="./pretrained/Metface"
for pkl in `find "$NETWORK_DIR" -name "*.pkl" -type f`
do
  python calc_metric.py --gpus 4 --metrics $METRIC --network $pkl --pkl_range $PKL_RANGE
done
python calc_metric.py --gpus 4 --metrics $METRIC --network $pkl --find_best True