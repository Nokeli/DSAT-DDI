for i in {0..4}; do
    python train.py --fold $i > "drugbank_fold619_$i.log" 2>&1
done