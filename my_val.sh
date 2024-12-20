python main.py \
    --data_root data \
    --val_data_path val_data \
    --prefix adaface_ir50_ms1mv2 \
    --gpus 1 \
    --use_16bit \
    --start_from_model_statedict ./pretrained/adaface_ir50_ms1mv2.ckpt \
    --arch ir_50 \
    --evaluate