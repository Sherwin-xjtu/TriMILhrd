#EMDD

Implementation of the EMDD algorithm in Python. Syntax: `python3 emdd.py <data-set-name>`. `<data-set-name>` is one of the following:

 - `fake`: Clearly-separable fake-data that demonstrates the correctness of the implementation.
 - `musk1`: MUSK1 data.
 - `musk2`: MUSK2 data.
 - `synth1`: Synthetic data 1 (provided)
 - `synth4`: Synthetic data 4 (provided)
 - `dr`: Diabetic-Retinopathy data.
 - `elephant`: Elephant data-set.
 - `fox`: Fox data-set.
 - `tiger`: Tiger data-set.

The script will perform 10-fold cross-validation on the supplied data-set and report the average accuracy, precision, and recall. It will also attempt to run the classifier on the entire data-set using the average target and scale of 10-fold cross-validation runs. The accuracy, precision, and recall are also reported in this case.
