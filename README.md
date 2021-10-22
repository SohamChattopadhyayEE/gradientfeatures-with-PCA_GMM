# Gradient features extraction and aggregation with PCA and GMM
## Requirements
- `python 3.6 or above`
- `pip install numpy`
- `pip install scikit-image`
- `pip install scikit-learn`
- `pip install opencv-python`
## Code execution
  ### Feature extraction : 
  - run the code [feature_extraction.py](https://github.com/SohamChattopadhyayEE/gradientfeatures-with-PCA_GMM/blob/main/Codes/feature_extraction.py) with adequate data path and file path.
  - The gradient features will be saved as `.npy` file named as `features.npy` at mentioend path.
  ### Fishervector model :
  - run the code [pcagmm.py](https://github.com/SohamChattopadhyayEE/gradientfeatures-with-PCA_GMM/blob/main/Codes/pcagmm.py) to implement Fishervector model.
  ### Input:
  - path of the `features.npy` file.
  - output path where the files are to be saved.
  ### Output:
  - `mean.npy` - fisher vector mean with dimension kd x nc.
  - `cov.npy` - fisher vector covariance with dimention kd x nc. 
  - `fv.npy` - fisher vector
