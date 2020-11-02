
# Fixed-point Convolution Framework

## Scirpts

We provide three Python scripts with the following functions:
The fixed-point model is decomposed into integer models and corresponding scaling factors.
+ [transform_model.py](./transform_model.py): decomposing the fixed-point model (lrs_fixed.pth) into integer model (lrs_fixed_real.pth) and corresponding scaling factors (range.npy).
+ [test_single.py](./test_single.py): testing model on one image.
+ [test_kodak.py](./test_kodak.py): testing model on kodak set.

## Models

We provide two models, which are generated as follows:

+ [lrs_float.pth](./lrs_float.pth): floating-point model.
+ [lrs_fixed.pth](./lrs_fixed.pth): fixed-point model that will be decomposed into range.npy and lrs_fixed_real.pth.


## Steps

The two Python scripts we provide are used as follows:

+ python transform_model.py
+ python test_single.py or python test_kodak.py

+ [lrs_fixed_real.pth](./lrs_fixed_real.pth): fixed-point model whose weights and biases are integers


## Result

![Result](./result.png)

Value of lrs_fixed_real.pth to be shown

**TODO**
