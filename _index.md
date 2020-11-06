
## Scirpts

We provide three Python scripts with the following functions:

+ [transform_model.py](./scripts/transform_model.py): decomposing the **fixed-point model** (lrs_fixed.pth) into **integer model** (lrs_integer.pth) and corresponding **range factors** (range.npy).
+ [test_single.py](./scripts/test_single.py): testing model on one image.
+ [test_kodak.py](./scripts/test_kodak.py): testing model on Kodak dataset.

## Models

We provide two models, which are generated as follows:

+ [lrs_float.pth](./models/lrs_float.pth): floating-point model.
+ [lrs_fixed.pth](./models/lrs_fixed.pth): fixed-point model that will be decomposed into range.npy and lrs_integer.pth.


## Steps

The two Python scripts we provide are used as follows:

+ python transform_model.py
+ python test_single.py or python test_kodak.py

+ [lrs_integer.pth](./models/lrs_integer.pth): fixed-point model whose weights and biases are integers.


**TODO**

The calculation process is shown in the figure below:

<img src="https://njuvision.github.io/fixed-point/images/framework.png" width="400px" >
