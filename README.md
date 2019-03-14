# equalization_code

This code is an implementation example of the equalization algorithm as introduce in the paper 

Same, Same But Different: Recovering Neural Network Quantization Error
Through Weight Factorization.

To use this repository you need to install git LFS. Follow the instructions here: https://git-lfs.github.com/.

After cloning the reposetory install the requirements.
 
```bash
pip install -r requirements
```

There are 5 examples: ResNet_v1_18, Inception_v1, Inception_v3, DenseNet121, and MobileNet_v2_1.4. For each net there are net description file <.hn>, weights file <_params.npz>, and extremum values per layer file <*_layer_inference_data.npz>. The net description files are "caffe like" txt files that describe the network graph. The extermum values are obtained by running a calibration batch of 64 images through the network and, for each layer, taking the min,max value of the activation as the extremum value (note: this is done after the BN layer has been folded). 

The algorithm described in the paper is fully contained within equlization.py. The default setting is to run the one_step method and is run by:

```bash
python equalization.py
```
Two step algorithm can be run by calling 
```bash
python equalization.py two_steps
```

The script will generate post equalization weights and store them to files <*._params_eq.npz>





