import h2o
import pandas as pd
import numpy as np
import math


# Starts a local H2O node
h2o.init()

# Read training data into memory
iotRaw = pd.read_csv('BH11D_labelled.csv')

# Select training columns

iot = iotRaw.iloc[:,[1,2,3,4,5]]
iot_y = iotRaw.iloc[:,6]
total_size=len(iot)
train_size=math.floor(0.70*total_size) 

iot_np = iot.values

#training dataset
X_train=iot.head(train_size)
y_train = iot_y.head(train_size)
#test dataset
X_test=iot.tail(len(iot) -train_size)
y_test = iot_y.tail(len(iot_y) - train_size)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

# Send the data to H2O
iot_hex_train  = h2o.H2OFrame(X_train)
iot_hex_test = h2o.H2OFrame(X_test)


# Run the deeplearning model in autoencoder model
iot_dl = h2o.estimators.deeplearning.H2OAutoEncoderEstimator(model_id = "iot_dl",
                                                             autoencoder = True,
                                                             hidden=[50, 20, 2, 20, 50],
                                                             epochs = 100,
                                                             l1 = 1e-5, l2 = 1e-5, max_w2 = 10.0,
                                                             activation = "TanhWithDropout",
                                                             initial_weight_distribution = "UniformAdaptive",
                                                             adaptive_rate = True)

iot_dl.train(
    x=iot_hex_train.names,
    training_frame=iot_hex_train
)

iot_dl.get_params()




# Make predictions for training data
iot_error = iot_dl.anomaly(iot_hex_test)

iot_error_df = iot_error.as_data_frame()
iot_error_np = iot_error_df.values

# Threshold Function for outliers: Z-Score Method
def outliers_z_score(ys):
    threshold = 3

    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    return np.where(np.abs(z_scores) > threshold)

outliers_z_score_np = outliers_z_score(iot_error_np)


# Threshold Function for Outliers: Modified Z-Score Method

def outliers_modified_z_score(ys):
    threshold = 3.5

    median_y = np.median(ys)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
                         for y in ys]
    return np.where(np.abs(modified_z_scores) > threshold)

outliers_modified_z_score_np = outliers_modified_z_score(iot_error_np)


# Threshold Function for Outliers: Modified Z-Score Method
def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))

outliers_iqr_np = outliers_iqr(iot_error_np)

original_count = np.count_nonzero(y_test)
z_score_array = np.asarray(outliers_z_score_np[0])
outliers_modified_z_score_array = np.asarray(outliers_modified_z_score_np[0])
outliers_iqr_np_array = np.asarray(outliers_iqr_np[0])
