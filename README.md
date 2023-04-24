# torch_mvs

## Installing
```sh
python setup.py develop
```

## Preprocessing
```sh
python -m tmvs.colmap2mvs --dense_folder $DATA_DIR/dense/ --save_folder $RESULT_ROOT
```

## Extract superpixels
```sh
python -m tmvs.extract_superpixel --img_dir $DATA_DIR/images --save_dir $SUPERPIXEL_DIR
```

## Running
```sh
python -m tmvs.launch -rf $RESULT_ROOT # --suffix $MVS_SUFFIX)(optional)
```

## Postprocessing
```sh
python -m tmvs.mvs_fusion_segmentaion --depth_normal_dir $MVS_DIR/depth_normal/ \
        --data_dir $DATA_DIR --superpixel_dir $SUPERPIXEL_DIR/ \
        --save_dir $MVS_DIR/planar_prior/ --vis --clean_mesh # (--gen_mask --mask_suffix planar_mask_mvs) for init 
```
