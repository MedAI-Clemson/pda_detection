This is the official Pytorch implementation of the paper [On the Relevance of Temporal Features for Medical Ultrasound Video
  Recognition](https://link.springer.com/chapter/10.1007/978-3-031-43895-0_70) by D. Hudson Smith, John Paul Lineberger, and George H. Baker.

# Abstract
Many medical ultrasound video recognition tasks involve identifying key anatomical features regardless of when they appear in the video suggesting that modeling such tasks may not benefit from temporal features. Correspondingly, model architectures that exclude temporal features may have better sample efficiency. We propose a novel multi-head attention architecture that incorporates these hypotheses as inductive priors to achieve better sample efficiency on common ultrasound tasks. We compare the performance of our architecture to an efficient 3D CNN video recognition model in two settings: one where we expect not to require temporal features and one where we do. In the former setting, our model outperforms the 3D CNN - especially when we artificially limit the training data. In the latter, the outcome reverses. These results suggest that expressive time-independent models may be more effective than state-of-the-art video recognition models for some common ultrasound tasks in the low-data regime.

# Citation
If you use this work, please cite
```
@inproceedings{smith2023relevance,
  title={On the Relevance of Temporal Features for Medical Ultrasound Video Recognition},
  author={Smith, D Hudson and Lineberger, John Paul and Baker, George H},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={744--753},
  year={2023},
  organization={Springer}
}
```

# Paper version
The published paper is based on the following commit: https://github.com/MedAI-Clemson/pda_detection/tree/1962cfcfe44dbe18f9ad7383e5a898b7859c95a0 

# Data

# Project structure
 
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

# Usage
