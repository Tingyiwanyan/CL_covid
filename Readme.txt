This code implement Contrastive learning for covid event prediction
the original data is not provided due to the hospital policy.

To run the code, the data set and corresponding label should be provided.

The data should be in LxNxM numpy array format, where L is the length of the data, N is the time sequence dimension, M is the feature length dimension.

the hyper-parameters also need to be configured in CL_prediction.py, like time sequence length for LSTM model, batch size and latent feature dimensions.

To run the code, just run main.py with data and label provided. 
