# Matrix Capsules with EM Routing
Re-implement the CapsNet described in "[Matrix Capsules with EM Routing](https://openreview.net/pdf?id=HJWLfGWRb)".

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You will need `python3.5` and

* `numpy` == 1.13.3
* `tensorflow` (`tensorflow-gpu`) >= 1.4
* `tqdm` == 4.19.2

Our experiments are run on `Ubuntu 16.04.2 LTS` with kernel version `4.4.0-97-generic`, CPU `E5-2640 v4`, RAM 94.2G and GPU GTX 1080. But there should be no specific system requirement, except an NVIDIA graphic card if you want GPU acceleration.

### Installing
Nothing specific to install.

### Prepare the dataset

Our script only accepts npz file format. So to play with SmallNORB, some conversion is required.

First you need to download script from https://github.com/ndrplz/small_norb/blob/master/smallnorb/dataset.py to folder `scripts`, then download [SmallNORB dataset](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/) to some folder. Then run

```
python smallnorb2npz.py [SmallNORB-bin] [SmallNORB-npz]
```

where 

- `[SmallNORB-bin]` is the folder containing the SmallNORB binary files downloaded from [website of SmallNORB](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/). (Including `smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat`, `smallnorb-5x46789x9x18x6x2x96x96-training-info.mat`, `smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat`, `smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat`, `smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat`, and `smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat`, totally 6 files)
- `[SmallNORB-npz]` is the folder where the converted npz file will be outputed.

## Training

Assuming the dataset is located under folder `data` and the results as well as checkpoints will be stored under `logs`
```
python src/train.py data logs
```
batch size is default to 128.

You can specify which model architecture to train and which loss to use with parameters `--arch` and `--loss`. For example, to train a CNN network with spread loss, you can specify `--arch cnn --loss spread_loss`. You can use `--help` to see more information.

## Experiment with Adversarial Attack

To experiment with adversarial attack, please run script `src/adversarial_attack.py`. For example, if you want to see how CNN with spread loss works under BIM attack, you can run:

```
python3 adversarial_attack.py ./data/SmallNORB/ ./models/cnn-spread/ adversarial-attack-BIM.log --arch cnn --loss spread --method BIM
```

Then the accuracy for different epsilon can be found in the file `adversarial-attack-BIM.log`.

## Authors
* **Tin-Ray Chiang** - *Initial work* - [CTinRay](https://github.com/CTinRay)
* **Yen-Ting Liu** - Clean up, filter visualization, benchmarks - [liuyenting](https://github.com/liuyenting)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments
* [Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf)
* [Matrix Capsules with EM Routing](https://openreview.net/pdf?id=HJWLfGWRb)
