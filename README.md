# S2-Unrolling
Pytorch codes for the paper "Unsupervised Sentinel-2 Image Fusion Using A Deep Unrolling Method", submitted in IEEE GRSL. <br>
**Authors:** Han V. Nguyen $^\ast \dagger$, Magnus O. Ulfarsson $^\ast$,  Johannes R. Sveinsson $^\ast$, and Mauro Dalla Mura $^\ddagger$ <br>
$^\ast$ Faculty of Electrical and Computer Engineering, University of Iceland, Reykjavik, Iceland<br>
$^\dagger$ Department of Electrical and Electronic Engineering, Nha Trang University, Khanh Hoa, Vietnam<br>
$^\ddagger$ GIPSA-Lab, Grenoble Institute of Technology, Saint Martin d’Hères, France.<br>
Email: hvn2@hi.is

## Abstract:<br>
Multispectral remote sensing images are often available in multiple resolutions due to cost and technical limitations. To address this, we developed a method that sharpens low-resolution (LR) images using high-resolution (HR) images. In this paper, we propose a novel unsupervised deep learning (DL) approach that involves unrolling an iterative algorithm into a deep neural network and training it using a loss function based on Stein's risk unbiased estimate (SURE) to sharpen the LR bands (20 and 60 m) of Sentinel-2 (S2) to their highest resolution (10 m). This approach views traditional optimization model-based methods through a DL framework, improving interpretability and clarifying connections between the two approaches. Results from both simulated and real S2 datasets demonstrate that the proposed method outperforms competitive methods and produces high-quality images for the 20 m and 60 m bands. The codes will be made available at: https://github.com/hvn2/S2-Unrolling. <br><br>
 **Please cite our paper if you are interested**<br>
 @inproceedings{nguyen2023s2_unroll,
  title={Unsupervised Sentinel-2 Image Fusion Using A Deep Unrolling Method},
  author={Nguyen, Han V and Ulfarsson, Magnus O and Sveinsson, Johannes R and Dalla Mura, Mauro},
  booktitle={IEEE Geoscience and Remote Sensing Letters},
  pages={0--0},
  year={2023},
  organization={IEEE}
}
## Usage:<br>
The following folders contanin:
- data: The simulated dataset.
- models: python scripts define the model (network structure)
- utils: additional functions<br>
**Run the jupyter notebook and see results.**
## Environment
- Pytorch 1.8
- Numpy, Scipy, Skimage.
- Matplotlib