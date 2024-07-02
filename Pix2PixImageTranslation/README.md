# Pix2Pix Image Translation

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Environment Setup
- Clone this repo and change directory
  ```
  https://github.com/nmazda/BubbleProject.git
  cd BubbleProject/Pix2PixImageTranslation
  ```
- Install [PyTorch](http://pytorch.org) and 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate))
- For pip users, please type the command `pip install -r requirements.txt`
- For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`

## Prepare Dataset 

#### Download dataset :
- Download a dataset and save to datasets folder with subdirectories `A` and `B`, for example:
```bash
./datasets/BubbleData        # /path/to/data
./datasets/BubbleData/A      # /path/to/data/A
./datasets/BubbleData/B      # /path/to/data/B
```
- `A` and `B` should each have their own subdirectories `train`, `val`, `test`, etc. In `/path/to/data/A/train`, put training images in style A. In `/path/to/data/B/train`, put the corresponding images in style B. Repeat same for other data splits (`val`, `test`, etc)

#### Process dataset for pix2pix :
- Pix2pix's training requires paired data. We provide a python script to generate training data in the form of pairs of images {A,B}, where A and B are two different depictions of the same underlying scene. For example, these might be pairs {label map, photo} or {bw image, color image}. 
- Create folder `/path/to/data` with subdirectories `A` and `B`. `A` and `B` should each have their own subdirectories `train`, `val`, `test`, etc. In `/path/to/data/A/train`, put training images in style A. In `/path/to/data/B/train`, put the corresponding images in style B. Repeat same for other data splits (`val`, `test`, etc).
- Corresponding images in a pair {A,B} must be the same size and have the same filename, e.g., `/path/to/data/A/train/1.jpg` is considered to correspond to `/path/to/data/B/train/1.jpg`.

- Once the data is formatted this way, call :
```bash
python datasets/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data
```

This will combine each pair of images (A,B) into a single image file, ready for training.

## Training & Testing 

#### Train the model:
- To run the experiment multiple times use following bash script
```bash
chmod +x train_pix2pix.sh 
./train_pix2pix.sh --dataroot ./datasets/BubbleData --name bubble_pix2pix --runs 10
```
- To directly run the python file use following command
```bash
python train.py --dataroot ./datasets/BubbleData --name bubble_pix2pix --model pix2pix --direction AtoB
```

- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- To log training progress and test images to W&B dashboard, set the `--use_wandb` flag with train and test script
- To see more intermediate results, check out  `./checkpoints/bubble_pix2pix/web/index.html`.

#### Test the model :
```bash
python test.py --dataroot ./datasets/BubbleData --name bubble_pix2pix --model pix2pix --direction AtoB
```
- The test results will be saved to a html file here: `./results/bubble_pix2pix/test_latest/index.html`.
