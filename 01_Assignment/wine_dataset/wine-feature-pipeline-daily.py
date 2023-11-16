import os



def generate_wine(type, fixed_acidity, volatile_acidity, citric_acid,
       residual_sugar, chlorides, free_sulfur_dioxide,
       total_sulfur_dioxide, density, pH, sulphates, alcohol,
       quality):
    """
    Returns a single wine as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df = pd.DataFrame({ #"type": [random.uniform(sepal_len_max, sepal_len_min)],
                        "type": type,
                       "fixed_acidity": fixed_acidity,
                       "volatile_acidity": volatile_acidity,
                       "citric_acid": citric_acid,
                       "residual_sugar" : residual_sugar,
                       "chlorides" : chlorides,
                       "free_sulfur_dioxide" : free_sulfur_dioxide,
                       "total_sulfur_dioxide" : total_sulfur_dioxide,
                       "density" : density,
                       "pH" : pH,
                       "sulphates" : sulphates,
                       "alcohol" : alcohol,
                      },index=[0])
    df['quality'] = quality
    return df


def get_random_wine():
    """
    Returns a DataFrame containing one random wine
    """
    import pandas as pd
    import random

    wine01_df = generate_wine("white", 7.1, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.0010, 3.0, 0.45, 8.8, 6)
    wine02_df = generate_wine("white", 7.2, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.0010, 3.0, 0.45, 8.8, 6)
    wine03_df =  generate_wine("white", 7.3, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.0010, 3.0, 0.45, 8.8, 6)

    # randomly pick one of these 3 and write it to the featurestore
    pick_random = random.uniform(0,3)
    if pick_random >= 2:
        iris_df = wine01_df
        print("Wine 01 added")
    elif pick_random >= 1:
        iris_df = wine02_df
        print("Wine 02 added")
    else:
        iris_df = wine03_df
        print("Wine 03 added")

    return iris_df

def get_sample_from_distribution_wine(): 
    """
    Returns a DataFrame containing one random wine based on the feature distribution
    """
    import numpy as np
    import pandas as pd

    # Define your mean and std for each column
    features_distribution = {
    'fixed_acidity': [6.974958694754234, 0.8783412896705036],
    'volatile_acidity': [0.2989859562164395, 0.11510431133596595],
    'citric_acid': [0.3158364312267658, 0.10104684699102408],
    'residual_sugar': [5.557362660057827, 4.500478870551202],
    'chlorides': [0.04776352746798844, 0.016409034138339804],
    'free_sulfur_dioxide': [31.842110698058654, 15.644928810505371],
    'total_sulfur_dioxide': [124.81990912845932, 50.17105209423161],
    'density': [0.9941852767451467, 0.0027659851769307286],
    'ph': [3.2100681536555142, 0.14793746725556323],
    'sulphates': [0.5058715406856671, 0.11572175643265356],
    'alcohol': [10.532072834912846, 1.1741406701134156],
    'quality': [5.821354812061132, 0.7651886817920075]
    }

    # For numerical columns, use the specified mean and std to sample from a Gaussian distribution
    new_numerical_data = {column: np.random.normal(mean, std) for column, (mean, std) in features_distribution.items()}

    # For the categorical 'type' column, you need to provide the distribution. 
    type_distribution = {'white': 0.858736, 'red': 0.141264}  # Replace with actual distribution
    new_categorical_data = {'type': np.random.choice(list(type_distribution.keys()), p=list(type_distribution.values()))}

    # Combine both dictionaries into one data point
    new_data_point = {**new_numerical_data, **new_categorical_data}

    # Convert the data point into a DataFrame
    new_data_df = pd.DataFrame([new_data_point])

    # Display the new data point
    #print(new_data_df)
    return new_data_df



def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    iris_df = get_sample_from_distribution_wine()

    iris_fg = fs.get_feature_group(name="wine",version=3)
    iris_fg.insert(iris_df)

if __name__ == "__main__":
    g()
