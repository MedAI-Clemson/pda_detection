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