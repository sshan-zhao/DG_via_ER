## The PyTorch implemention of our work on domain generalization: 

S. Zhao, M. Gong, T. Liu, H. Fu and D. Tao. Domain Generalization via Entropy Regularization. NeurIPS 2020. 

### Environment
* Python 3.6
* PyTorch 1.1.0

### Run

* Download the PACS dataset (https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk), and prepare the dataset as follows:
    ```
    |- dataset
        |- PACS
            |- art_painting
                |- *.jpg
            |- cartoon
                |- *.jpg
            |- sketch
                |- *.jpg
            |- photo
                |- *.jpg
    ```
* Execute the following command to run (replace xxx with any one of the four datasets, such as art_painting):

    ```
    python train_PACS.py --target xxx
    ````
### Citation
```
@article{zhao2020domain,
  title={Domain Generalization via Entropy Regularization},
  author={Zhao, Shanshan and Gong, Mingming and Liu, Tongliang and Fu, Huan and Tao, Dacheng},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
### Contact

Shanshan Zhao: szha4333 AT uni dot sydney dot edu dot au / sshan.zhao00 AT gmail dot com
