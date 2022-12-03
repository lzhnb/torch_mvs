# torch_mvs

## Installing
```sh
python setup.py develop
```

## Preprocessing
```sh
python scripts/colmap2mvs.py --dense_folder $DATA_DIR/dense/ --save_folder $RESULT_ROOT
```

## Extract superpixels
```sh
python scripts/extract_superpixel.py --img_dir $DATA_DIR/images --save_dir $SUPERPIXEL_DIR
```

## Running
```sh
python -m tmvs.launch -rf $RESULT_ROOT # --suffix $MVS_SUFFIX)(optional)
```
