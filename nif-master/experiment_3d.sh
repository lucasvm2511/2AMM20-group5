date 

CONFIGURATION_PATH=$1
FILE_PATH=$2
RESULTS_ROOT=$3

mkdir -p $RESULTS_ROOT

COMPRESSED_PATH=$RESULTS_ROOT/compressed.nif
DECODED_PATH=$RESULTS_ROOT/decoded.npy
STATS_PATH=$RESULTS_ROOT/stats.json

python3 $HOME/nif-master/encode_3d.py $CONFIGURATION_PATH $FILE_PATH $COMPRESSED_PATH
python3 $HOME/nif-master/decode_3d.py $CONFIGURATION_PATH $COMPRESSED_PATH $DECODED_PATH

python3 $HOME/nif-master/filewise_export_stats_3d.py \
    $FILE_PATH \
    $DECODED_PATH \
    $STATS_PATH \
    $COMPRESSED_PATH

date
