


```bash
! git clone https://github.com/Morata91/AdaFace.git
%cd AdaFace
! pip install -r requirements.txt
! wget -P data/val_data http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz
! tar -xzvf data/val_data/lfw-funneled.tgz -C data/val_data
! python ex.py
! python main.py \
    --data_root data \
    --val_data_path val_data \
    --prefix adaface_ir50_ms1mv2 \
    --gpus 1 \
    --use_16bit \
    --start_from_model_statedict ./pretrained/adaface_ir50_ms1mv2.ckpt \
    --arch ir_50 \
    --evaluate
```


