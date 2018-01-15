# Matrix Capsules with EM Routing
Re-implement the CapsNet described in "[Matrix Capsules with EM Routing](https://openreview.net/pdf?id=HJWLfGWRb)".

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
You will need
```
tensorflow (tensorflow-gpu) >= 1.4
tqdm
```

### Installing
Nothing specific to install.

## Running the tests
**TODO** download smallNORM dataset using `scripts`  
Assuming the dataset is located under folder `data` and the results will be stored under `logs`
```
python src/train.py data logs
```
batch size is default to 128.

## Authors
* **Tin-Ray Chiang** - *Initial work* - [CTinRay](https://github.com/CTinRay)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments
* [Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf)
* [Matrix Capsules with EM Routing](https://openreview.net/pdf?id=HJWLfGWRb)
