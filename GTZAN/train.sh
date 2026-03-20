pip install -r requirements.txt
python train.py \
  --csv_path /Users/princelu/Desktop/NN/GTZAN/gtzan.csv \
  --audio_root /Users/princelu/Desktop/NN/GTZAN/genres \
  --out_dir /Users/princelu/Desktop/NN/GTZAN/checkpoints \
  --hidden_dim 256 \
  --num_layers 2 \
  --dropout 0.1 \
  --batch_size 64 \
  --epochs 100 \
  --validate_every 10 \
  --lr 1e-3 \
  --weight_decay 1e-4\
  --timeseries_length 128 \
  --hop_length 512 \
  --target_sr 22050