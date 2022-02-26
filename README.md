# Lung Nodule Detection in Pytorch
A One-stage Method for lung nodule detection in LUNA16 dataset

## Scripts:
- Would be executed:
  - config_training.py: set filepaths   
  - prepare.py: LUNA16 dataset preprocessing   
  - main_detector_recon.py: model training and inference
  - GenerateCSV.py: Generate result.csv for computing CPM
  - noduleCADEvaluaionLUNA16.py: Compute CPM of .csv
  - FROC_CPM.ipynb: Plot FROC curve
  
- Others
  - data_detector.py: generate data loader during training and testing (super difficult to undetstand)
  - preprocess.py: some preproceesing-related codes
  - layers.py
  - loss.py
  - split_combine.py (At Testing stage)
  - utils.py

## Requirements:
- Python 3.6
- torch 0.4.1
- torchvision 0.2.0
- SimpleITK
- scikit-image

## Files:
- LUNA.json: Every Case ID stored in .json for make_dataset.py
- test_0222_1 ~ test_0222_5: Five dir containing train/val/test.json (Case ID) respectively
- Download LUNA16 dataset from Grand Challeng: https://luna16.grand-challenge.org and save at the following filepaths  
  ./data/LUNA16/allset: all .raw and .mhd of LUNA16 data  
  ./data/LUNA16/seg-lungs-LUNA16: all .zraw and .mhd of LUNA16 mask

## How to Do step by step:
- Preprocessing for LUNA16
  - python prepare.py
  - output file path: config_training -> config[preprocess_result_path]
  ```
    Output: id_clean.npy & id_label.npy (for training) ; id_extendbox.npy & id_mask.npy & id_origin.npy & id_spacing.npy (for vox2world) 
  ```
- Start training and testing 
  - training
  ```
    python main_detector_recon.py --model OSAF_YOLOv3 -b [batch_size] --epochs [num_epochs] --save-dir [save_dir_path] --save-freq [save_freq_ckpt] --gpu '0' --n_test [number of gpu for test] --lr [lr_rate] --cross [1-5 set which cross_data be used] #--resume [resume ckpt]
    eg: python main_detector_recon.py --model OSAF_YOLOv3 -b 2 --epochs 100 --save-dir OSAF_YOLOv3_testcross1 --save-freq 1 --gpu '0' --n_test 1 --lr 0.001 --cross 1 
  ```
  - testing
  ```
    python main_detector_recon.py --model OSAF_YOLOv3 --resume [resume_ckpt] --save-dir [] --test 1 --gpu '0' --n_test [] --cross []
    eg: python main_detector_recon.py --model OSAF_YOLOv3 --test 1 --cross 1 --resume "./results/OSAF_YOLOv3_testcross1/195.ckpt" --save-dir "OSAF_YOLOv3_testcross1" --gpu 0
  ```
  ```
    output id_lbb.npy (label), id_pbb.npy (predicted bboxes)
  ```

- Compute CPM (After test all 5 fold)
  - Generate result.csv
  ```
    python GenerateCSV.py
  ```
  ```
    output: OSAF_YOLOv3_80_all.csv
  ```
  - Then compute CPM and save related .png and .npy 
  ```
    python noduleCADEvaluaionLUNA16.py (Remember to modify the filepath in noduleCADEvaluation.py)
  ```
  ```
    output: print(csv_name, CPM, seven_sensitivities@predefined_fps) and save ./CPM_Results_OSAF_YOLOv3_80/_.npy & _.png
  ```

- Plot FROC Curve
  ```
    Execute FROC_CPM.ipynb
  ```
  ```
    output: FROC_OSAF_YOLOv3.png
  ```

- How to transform voxel coord of pbb into world voxel of 3D CT ?
  - You can refer to GenerateCSV.py: How to transform pbb -> pos
