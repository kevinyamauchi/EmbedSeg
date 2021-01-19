## EmbedSeg 

### Introduction
This repository hosts the version of the code used for the **[publication]()** **Embedding-based Instance Segmentation of Microscopy Images**. For a short summary of the main attributes of the publication, please check out the **[project webpage]()**.

We refer to the techniques elaborated in the publication, here as **EmbedSeg**. `EmbedSeg` is a method to perform instance-segmentation of objects in microscopy images, and extends the formulation of **[Neven et al, 2019](https://arxiv.org/abs/1906.11109)**. 

<p float="left">
  <img src="https://mlbyml.github.io/EmbedSeg_RC/images/teaser/X_9_image_painted.gif" width="100" />
  <img src="https://mlbyml.github.io/EmbedSeg_RC/images/teaser/X_9_GT_painted.gif" width="100" /> 
  <img src="https://mlbyml.github.io/EmbedSeg_RC/images/teaser/X_9_painted.gif" width="100" />
</p>


In `EmbedSeg`, we suggest two simple tricks: by embedding interior pixels to instance medoids instead of the instance centroids and by including test-time augmentation during inference, we obtain state-of-the-art results on several datasets. Additionally by accumulating gradients over multiple steps, we allow our overall pipeline to have a small enough memory footprint to allow network training on virtually all CUDA enabled laptop hardware.

### Citation
If you find our work useful in your research, please consider citing:

```bibtex
@article{2021:EmbedSeg,
  title={Embedding-based Instance Segmentation of Microscopy Images},
  author={Lalit, Manan and Tomancak, Pavel and Jug, Florian},
  journal={},
  year={2021}
}
```

### Dependencies 
We have tested this implementation using `pytorch` version 1.1.0 and `cudatoolkit` version 10.0 on a `linux` OS machine. 

In order to replicate results mentioned in the publication, one could use the same virtual environment (`EmbedSeg_environment.yml`) as used by us. Create a new environment, for example,  by entering the python command in the terminal `conda env create -f path/to/EmbedSeg_environment.yml`.

### Getting Started

Please open a new terminal window and run the following commands one after the other.

```shell
git clone https://github.com/juglab/EmbedSeg.git
cd EmbedSeg
git checkout v0.1-basel-dataset
conda env create -f EmbedSeg_environment.yml
conda activate EmbedSegEnv
python3 -m pip install -e .
python3 -m ipykernel install --user --name EmbedSegEnv --display-name "EmbedSegEnv"
cd examples/2d
jupyter notebook
```

Look in the `examples` directory. If you would like to train the model from scratch, use `02-train.ipynb`. Otherwise, you could evaluate on unseen eimages using `03-predict.ipynb` and the trained model weights available [here](). Please make sure to select `Kernel > Change kernel` to `EmbedSegEnv` before running the notebooks. 


### Results

10 % of the available data was reserved for evaluation. In terms of the Average Precision [AP](https://cocodataset.org/#detection-eval) metric at different IOU thresholds, the model trained for ~40 epochs provided the following results:

| Threshold | AP<sub>50</sub> | AP<sub>55</sub>| AP<sub>60</sub> | AP<sub>65</sub>| AP<sub>70</sub>| AP<sub>75</sub> | AP<sub>80</sub> | AP<sub>85</sub> | AP<sub>90</sub>
|-	|-	|-	|-	|-	|- | -| - | -| -|	
| | 0.933| 0.930 | 0.927| 0.923 | 0.917 | 0.900 | 0.847 | 0.755 |0.614


Here below is the training and validation loss curve:

<img src="https://mlbyml.github.io/EmbedSeg_RC/images/teaser/loss.png" width="500" />



### Inference on unseen data
   
Use the **[03-predict.ipynb](https://github.com/juglab/EmbedSeg/blob/v0.1-basel-dataset/examples/2d/basel-2020/03-predict.ipynb)** notebook, have a look at the last four code cells especially! 
To predict on a directory containing images, one could use *pseudo code* such as the following: 

```python3
from glob import glob
import tifffile
import os
# input_dir_name = 'path-to-directory-containing-images'
# output_dir_name = 'path-to-directory-where-predictions-should-be-saved'
image_file_names = glob(input_dir_name/*.tif)
for image_file_name in image_file_names:
  im = tifffile.imread(image_name.tif)
  # load model
  # move numpy-array image to GPU
  # pad image so that its height and width are multiples of 8
  output = model(im_z)
  instance_map, predictions = cluster.cluster(prediction = output[0])
  tifffile.imsave(os.path.join(output_dir_name, os.path.basename(image_name)), instance_map)  
```

