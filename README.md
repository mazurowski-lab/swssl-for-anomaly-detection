# SWSSL - Sliding window-based self-supervised learning for anomaly detection in high-resolution images  (IEEE Trans. on Medical Imaging 2023)

#### By Haoyu Dong, Yifan Zhang, Hanxue Gu, [Nicholas Konz](https://nickk124.github.io/), Yixin Zhang, and [Maciej Mazurowski](https://sites.duke.edu/mazurowski/).

[Paper Link here](https://ieeexplore.ieee.org/abstract/document/10247020)

This is the official repository for our image anomaly detection model **SWSSL** in [IEEE Trans. on Medical Imaging 2023](https://ieeexplore.ieee.org/abstract/document/10247020). In this paper, we extend anomaly detection to high-resolution images by proposing to train the network and perform inference at the **patch level**, through the sliding window algorithm. We further study the augmentation function in the context of medical imaging when learning augmentation-invariant features. In particular, we observe that the **resizing** operation, a key augmentation in general computer vision literature, is detrimental to detection accuracy, and the **inverting** operation can be beneficial. We also propose a new module that encourages the network to learn from adjacent patches to boost detection performance.

## Model training and evaluation
### Step 0: Data Acquisition:

You can obtain the Chest XRay dataset from [here](https://data.mendeley.com/datasets/rscbjbr9sj/2), and the DBT dataset from [here](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=64685580). We are also happy to provide pre-processed data upon request.

### Step 1: Model Running 

You can run "bash run_chest.sh" to obtain the performance reported in the paper. Running on the DBT dataset is very similar, but you need to change some hyperparameters as described in the paper.

## Citation

Please cite our paper if you use our code or reference our work (published version citation forthcoming):
```bib
@ARTICLE{10247020,
  author={Dong, Haoyu and Zhang, Yifan and Gu, Hanxue and Konz, Nicholas and Zhang, Yixin and Mazurowski, Maciej A},
  journal={IEEE Transactions on Medical Imaging}, 
  title={SWSSL: Sliding window-based self-supervised learning for anomaly detection in high-resolution images}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2023.3314318}}
```
