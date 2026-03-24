python predict_trans.py \
  --csv_path /Users/princelu/Desktop/NN/GTZAN/gtzan.csv \
  --audio_root /Users/princelu/Desktop/NN/GTZAN/genres \
  --ckpt_path /Users/princelu/Desktop/NN/GTZAN/checkpoints_transformer_logmel/checkpoint_best.pt \
  --out_csv /Users/princelu/Desktop/NN/GTZAN/61447070S_submission_trans.csv \
  --num_layers 2 \
  --dropout 0.1 \
  --timeseries_length 256 \
  --hop_length 256 \
  --target_sr 22050
