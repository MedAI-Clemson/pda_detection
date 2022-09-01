# Environment setup
First create a new conda environment
```bash
conda create -n pda python=3.9
```

Activate it
```bash
source activate pda
```

Install the necessary dependencies
```bash
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

Create the jupyter kernel
```bash
python -m ipykernel install --user --name pda --display-name "PDA"
```

You should now be able to select the pda environment within jupyter.

# Code folder:

### Notebooks: 
* 0_preprocess_frames: used to generate image frames and supporting metadata. Only used when new label data becomes available. 
* 1_make_patient_study_table: used to generate additional patient metadata used to make train/test splits. To be run after 0_preprocess_frames if new label data becomes available.
* 2a_examine_images: test image data loading and preprocessing by looking at batches of image data
* 2b_examine_videos: test video data loading and preprocessing by looking at batches of video data
* 3a_model_frames: train image classifier to distinguish pda from non-pda frames
* 3a_model_frames-multitask: train image classifier to distinguish pda from non-pda frames as well as distinguish among views and modes
* 3b_model_videos: train video classifier to distinguish pda from non-pda clips
* 4a_evaluate: evaluate models trained using notebook 3a_model_frames
* 4a_evaluate-multitask: evaluate models trained using notebook 3a_model_frames-multitask. Note: incomplete
* 4b_evaluate_vid: evaluate models trained using notebook 3b_model_videos
* create_anonymous_patient_ids: used to make anonymous patient ids to avoid sharing personally identifiable information

### Python scripts:
* dataset.py: pytorch dataset classes used to load image and video data
* models.py: pytorch model classes for image or video classification
* transforms.py: pre-defined sets of pytorch transform for data preprocessing and augmentation