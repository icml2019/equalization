# equalization_code

This code is an implementation example of the equalization algorithm as introduce in the paper 

Same, Same But Different: Recovering Neural Network Quantization Error
Through Weight Factorization.

To run this code clone this repository, and install the requirements
 
```bash
pip install -r requirements
```

There are 5 examples: resNet_v1_18, inception_v1, inception_v3, denseNet121, and mobileNet_v2_1.4. To each net there are net description 
file <*.hn>, weights file <*_params.npz>, and extremum values per layer file <*_layer_inference_data.npz>  

you can run the code by simply run the equalization file


```bash
python equalization.py
```

The script will generate post equalization weights and store them to files <*._params_eq.npz>





