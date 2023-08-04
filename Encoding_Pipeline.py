import numpy as np
import pandas as pd
from typing import List

from category_encoders.ordinal import OrdinalEncoder
from category_encoders.woe import WOEEncoder
from category_encoders.target_encoder import TargetEncoder
from category_encoders.sum_coding import SumEncoder
from category_encoders.m_estimate import MEstimateEncoder
from category_encoders.backward_difference import BackwardDifferenceEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.helmert import HelmertEncoder
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.james_stein import JamesSteinEncoder
from category_encoders.one_hot import OneHotEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.binary import BinaryEncoder
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from scipy.stats import spearmanr
import scipy.sparse as sp

def get_single_encoder(encoder_name: str, cat_cols: list):
    """
    Get encoder by its name
    :param encoder_name: Name of desired encoder
    :param cat_cols: Cat columns for encoding
    :return: Categorical encoder
    """
    if encoder_name == "BinaryEncoder":
        encoder = BinaryEncoder(cols=cat_cols)

    if encoder_name == "LeaveOneOutEncoder":
        encoder = LeaveOneOutEncoder(cols=cat_cols)

    if encoder_name == "FrequencyEncoder":
        encoder = FrequencyEncoder(cols=cat_cols)

    if encoder_name == "WOEEncoder":
        encoder = WOEEncoder(cols=cat_cols)

    if encoder_name == "TargetEncoder":
        encoder = TargetEncoder(cols=cat_cols)

    if encoder_name == "SumEncoder":
        encoder = SumEncoder(cols=cat_cols)

    if encoder_name == "MEstimateEncoder":
        encoder = MEstimateEncoder(cols=cat_cols)

    if encoder_name == "LeaveOneOutEncoder":
        encoder = LeaveOneOutEncoder(cols=cat_cols)

    if encoder_name == "HelmertEncoder":
        encoder = HelmertEncoder(cols=cat_cols)

    if encoder_name == "BackwardDifferenceEncoder":
        encoder = BackwardDifferenceEncoder(cols=cat_cols)

    if encoder_name == "JamesSteinEncoder":
        encoder = JamesSteinEncoder(cols=cat_cols)

    if encoder_name == "OrdinalEncoder":
        encoder = OrdinalEncoder(cols=cat_cols)

    if encoder_name == "CatBoostEncoder":
        encoder = CatBoostEncoder(cols=cat_cols)

    if encoder_name == "MEstimateEncoder":
        encoder = MEstimateEncoder(cols=cat_cols)

    if encoder_name == "OneHotEncoder":
        encoder = OneHotEncoder(cols=cat_cols)
    return encoder

def process_cols(data):
    
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    categorical_data = data.loc[:, categorical_columns]
    numerical_data = data.drop(categorical_columns, axis=1)

    print("Categorical Columns:")
    print(categorical_data.head())

    print("\nNumerical Columns:")
    print(numerical_data.head())
    return categorical_data,numerical_data


def remove_high_missing_columns(data, threshold=0.6):
    # Calculate the percentage of missing values for each column
    missing_percentage = data.isnull().mean()

    # Select columns that have missing percentage less than or equal to the threshold
    selected_columns = missing_percentage[missing_percentage <= threshold].index

    # Drop columns with high missing values
    filtered_data = data[selected_columns]

    return filtered_data

def identify_nominal_ordinal(categorical_data, target_variable):
    """
    The function `identify_nominal_ordinal` identifies whether each column in a categorical dataset is
    nominal or ordinal based on the number of unique categories and the strength of correlation with a
    target variable.
    
    :param categorical_data: A pandas DataFrame containing the categorical data
    :param target_variable: The target_variable is the variable that you want to predict or analyze in
    relation to the categorical data. It is the variable that you want to determine the correlation with
    the categorical columns
    :return: a dictionary where the keys are the column names of the categorical data and the values are
    either "Binary", "Ordinal", or "Nominal" depending on the nature of the categorical variable.
    """
    categorical_info = {}

    for col in categorical_data.columns:
        categories = categorical_data[col].unique()
        num_categories = len(categories)

        if num_categories <= 2:
            categorical_info[col] = "Binary"
        else:
            # Check the strength of correlation between the categorical column and the target variable
            spearman_corr, _ = spearmanr(categorical_data[col], target_variable)
            is_ordinal = abs(spearman_corr) == 1.0
            categorical_info[col] = "Ordinal" if is_ordinal else "Nominal"

    return categorical_info

def is_interval_categorical_ordinal(data_column):
    unique_categories = data_column.unique()
    
    # Convert the categories to integers, keeping the order
    category_mapping = {category: i for i, category in enumerate(unique_categories)}
    mapped_data = data_column.map(category_mapping)
    
    # Calculate the differences between consecutive category mappings
    differences = mapped_data.diff().dropna()
    
    # Check if all differences are the same
    is_interval = all(differences == differences.iloc[0])
    
    return is_interval

def columns_with_low_cardinality(data, threshold=15):
    low_cardinality_columns = []
    high_cardinality_columns = []

    for col in data.columns:
        unique_values = data[col].nunique()
        if unique_values < threshold:
            low_cardinality_columns.append(col)
        else:
            high_cardinality_columns.append(col)

    return low_cardinality_columns,high_cardinality_columns


def estimate_sparse_matrix_memory_usage(columns, sample_data):
    memory_usages_mb = {}
    total_memory = 0

    # Calculate the number of rows and columns
    num_rows = len(sample_data)
    num_columns = len(columns)

    # Initialize a dictionary to store column indices and their respective non-zero element counts
    col_indices_nonzero_elements = {col: 0 for col in columns}

    # Iterate through the sample data to count non-zero elements for each column
    for col in columns:
        col_index = sample_data.columns.get_loc(col)
        nonzero_elements = np.count_nonzero(sample_data.iloc[:, col_index])
        col_indices_nonzero_elements[col] = nonzero_elements

    # Calculate the memory usage for each column and the total memory usage
    for col, num_nonzero_elements in col_indices_nonzero_elements.items():
        dtype_size = np.dtype(sample_data.dtypes[col]).itemsize
        memory_usage_bytes = num_nonzero_elements * dtype_size
        memory_usage_mb = memory_usage_bytes / (1024 * 1024)
        total_memory += memory_usage_mb
        memory_usages_mb[col] = memory_usage_mb

    print(memory_usages_mb)
    return total_memory





# ACTION ITEMS
# 1. seperate categorical columns ++
# 2. Ordinal or Nomninal columns
#     1. Ordinal
#         1. Interval data or equal interval values
#             1. ORDINAL / LABEL ENCODING
#         2. ASK FOR CONTRAST ENCODER
#             1. POLYNOMIAL, SUM ....
#             2. ORDIANL ENCODER as intermediate step
#                 -> CHECK CARDINALITY <15
#     2. Nominal
#         -> Check Cardinality
#         1. <15 
#             1.Decision tree based algorithm <--
#                 1. ---------------------------|------
#                 2. One hot encoding           |     |
#             2. lead to memory issues    <------     |
#                 1. Is some loss acceptable<---------|
#                     1.1. Binary encoding
#                     1.2. Feature hashing encodings
#                 2. Handle Overfitting
#                     1. Leave One Out Encoder
#                     2. TARGET, JAMES-STIEN ....
#      3.Binary
#      4.Merge Columns




# The below code is performing feature encoding on a dataset. It reads a CSV file, separates the
# target variable from the rest of the data, and then applies different encoding techniques to the
# categorical variables. It identifies whether each categorical variable is nominal or ordinal based
# on its correlation with the target variable. It then applies specific encoding techniques based on
# the type of variable. Finally, it combines all the encoded variables with the numerical variables to
# create a final encoded dataset.


if __name__ == "__main__":
    merge_list = []
    data = pd.read_csv("C:/CODE/JIO (JPL)/Feature_Encoding_Pipeline_AUTOML/data/adult.csv",sep=",")
    print("\nData => \n",data.head())
    print("\nColumn List : \n",data.columns.tolist())

    target = input("\nEnter Target Column = ")

    # Separate categorical and target variables

    non_target_data = data.drop(target, axis=1)
    target_col_data = data[target]

    #TODO : Apply enconding to target to make it numeric
    #^ Applying Ordianl Encoding
    enc = get_single_encoder("OrdinalEncoder", target)
    encoded_target_col = enc.fit_transform(target_col_data) #*
    print("\nEncoded target: ", encoded_target_col)

    # Seperating / Removing NaN columns
    non_target_data = remove_high_missing_columns(non_target_data)

    # Sperate Ctaegorical columns
    cat_col,num_col = process_cols(non_target_data)

    # Identify nominal or ordinal columns based on correlation with the target variable
    categorical_info = identify_nominal_ordinal(cat_col, target_col_data)

    nominal_col, ordinal_col, binary_col = [],[],[]

    # Output the result and seperate the columns
    print("\n\nCategorical Column Information:")
    for col, info in categorical_info.items():
        print(f"{col}: {info}")
        if(info == "Ordinal"):
            ordinal_col.append(col)
        elif(info == "Nominal"):
            nominal_col.append(col)
        else:
            binary_col.append(col)
    # print(ordinal_col)
    # print(nominal_col)
    # print(binary_col)

    ordinal_data = data[ordinal_col].copy()
    nominal_data = data[nominal_col].copy()
    binary_data = data[binary_col].copy()
    # print(ordinal_data.head())
    # print(nominal_data.head())
    # print(binary_data.head())

    interval_ordinal_col = []
    non_interval_ordinal_col = []
    encoded_ordinal_non_interval = []
    encoded_ordinal_interval = []
    
    #* TODO : CHECK FOR ORDINAL COLUMNS AND PROCESS IT

    for col in ordinal_data:
        if(is_interval_categorical_ordinal(col)):
            interval_ordinal_col.append(col)
        else:
            non_interval_ordinal_col.append(col)
    print("\nColumns with interval => \n",interval_ordinal_col)
    print("\nColumns wit no_interval => \n",non_interval_ordinal_col)

    #* TODO: APPLY ENCODINGS ON THE INTERVAL DATA
    #^ Applying Backward difference encoder
    if(len(interval_ordinal_col) != 0):
        enc = get_single_encoder("BackwardDifferenceEncoder", non_interval_ordinal_col)
        encoded_ordinal_non_interval = enc.fit_transform(data[non_interval_ordinal_col],encoded_target_col) #*
        print("\nBWD - ORDINAL_NON_INTERVAL ====> \n",encoded_ordinal_non_interval)  
        merge_list.append(encoded_ordinal_non_interval)

    #^ Applying Ordianl Encoding
    if(len(non_interval_ordinal_col) != 0):
        enc = get_single_encoder("OrdinalEncoder", interval_ordinal_col)
        encoded_ordinal_interval = enc.fit_transform(data[interval_ordinal_col],encoded_target_col) #*
        print("\nORDINAL - ORDINAL_INTERVAL ====> \n",encoded_ordinal_interval)
        merge_list.append(encoded_ordinal_interval)  

    # TODO: Ask for Contrast Encoders

    #! PROCESS NOMINAL DATA

    # Checking the cardinality and seperating the columns
    #*  TODO: Pass column data
    low_cardinality_col, high_cardinality_col = columns_with_low_cardinality(nominal_data)
    #high_cardinality_col = nominal_data.drop(low_cardinality_col, axis=1).columns.tolist()

    print("\nlow cardinality columns = ", low_cardinality_col)
    print("\nhigh cardinality columns = ", high_cardinality_col)

    #* TODO: Check for decision tree based algo
    if(len(low_cardinality_col) != 0):
        decision_tree_algo = input("\nIs the algorithm Tree Based (low cardinality) (yes/no) = ").lower()
        if(decision_tree_algo == "no"):
            # TODO: Apply One-Hot Encoding (low_cardinality)
            enc = get_single_encoder("OneHotEncoder", low_cardinality_col)
            encoded_low_cardinality = enc.fit_transform(data[low_cardinality_col],encoded_target_col) #*
            print("\n\nONE HOT Encoding - Low cardinality ====> \n",encoded_low_cardinality) 

        else:
            handle_overfitting = input("\nDo you want to enter overfiting (low cardinality) (yes/no) = ").lower()
            if(handle_overfitting == "yes"):
                #* TODO: Apply Leave-one-out Encoding
                enc = get_single_encoder("LeaveOneOutEncoder", low_cardinality_col)
                encoded_low_cardinality = enc.fit_transform(data[low_cardinality_col],encoded_target_col) #*
                print("\nLEAVE ONE OUT - Low cardinality ====> \n",encoded_low_cardinality)  
            else:
                #* TODO : Apply james-stien, target etc..
                enc = get_single_encoder("TargetEncoder", low_cardinality_col)
                encoded_low_cardinality = enc.fit_transform(data[low_cardinality_col],encoded_target_col) #*
                print("\nTARGET - Low cardinality ====> \n",encoded_low_cardinality)
        merge_list.append(encoded_low_cardinality)
        
    #& Processing High Cardinality data
    if(len(high_cardinality_col) != 0):
        memory_check = estimate_sparse_matrix_memory_usage(high_cardinality_col, nominal_data[high_cardinality_col])
        print("\n\nCalculating estimate sparse matrix memory usage....... ")
        print("\n\nMEMORY USED = ", memory_check)
        if(memory_check > 10):
            handle_overfitting = input("\nDo you want to enter overfiting (high cardinality) (yes/no) = ").lower()
            if(handle_overfitting == "yes"):
                #* TODO: Apply Leave-one-out Encoding
                enc = get_single_encoder("LeaveOneOutEncoder", high_cardinality_col)
                encoded_high_cardinality = enc.fit_transform(data[high_cardinality_col], encoded_target_col) #*
                print("\nLEAVE ONE OUT - high cardinality ====> \n",encoded_high_cardinality)
            else:
                #* TODO : Apply james-stien, target etc..
                enc = get_single_encoder("JamesSteinEncoder", high_cardinality_col)
                encoded_high_cardinality = enc.fit_transform(data[high_cardinality_col],encoded_target_col) #*
                print("\nJAMES-STIEN - high cardinality ====> \n",encoded_high_cardinality)
        else:
            decision_tree_algo = input("\nIs the algorithm Tree Based (high cardinality) (yes/no) = ").lower()
            if(decision_tree_algo == "no"):
                # TODO: Apply One-Hot Encoding (high_cardinality)
                enc = get_single_encoder("OneHotEncoder", high_cardinality_col)
                encoded_high_cardinality = enc.fit_transform(data[high_cardinality_col],encoded_target_col) #*
                print("\n\nONE HOT Encoding - High cardinality ====> \n",encoded_high_cardinality) 
                pass
            else:
                handle_overfitting = input("\nDo you want to enter overfiting (low cardinality) (yes/no) = ").lower()
                if(handle_overfitting == "yes"):
                    #* TODO: Apply Leave-one-out Encoding
                    enc = get_single_encoder("LeaveOneOutEncoder", high_cardinality_col)
                    encoded_high_cardinality = enc.fit_transform(data[high_cardinality_col],encoded_target_col) #*
                    print("\nLEAVE ONE OUT - High cardinality ====> \n",encoded_high_cardinality)  
                else:
                    #* TODO : Apply james-stien, target etc..
                    enc = get_single_encoder("TargetEncoder", high_cardinality_col)
                    encoded_high_cardinality = enc.fit_transform(data[high_cardinality_col],encoded_target_col) #*
                    print("\nTARGET ENC - High cardinality ====> \n",encoded_high_cardinality)
            
            merge_list.append(encoded_high_cardinality)
        
    #* Applying Binary encoding to Binary Columns
    if(len(binary_col) != 0):
        enc = get_single_encoder("BinaryEncoder", binary_col)
        encoded_binary_columns = enc.fit_transform(data[binary_col],encoded_target_col) #*
        merge_list.append(encoded_binary_columns)


    #& Combine all data for final output
    merge_list.append(num_col)
    result_df = pd.concat(merge_list, axis=1)
    print("\n\nFinal Encoded DataFrame = \n\n",result_df)













    



