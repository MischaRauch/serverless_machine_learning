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
    features_distribution_white = {
    'fixed_acidity': [6.852777777777778, 0.8007807437338395],
    'volatile_acidity': [0.2721693121693121, 0.08951306653042747],
    'citric_acid': [0.3232419432419432, 0.09302224974478986],
    'residual_sugar': [6.069708994708995, 4.634282967828375],
    'chlorides': [0.0430937950937951, 0.01145275534457891],
    'free_sulfur_dioxide': [34.25396825396825, 15.087977182325284],
    'total_sulfur_dioxide': [136.8893698893699, 41.42923547008297],
    'density': [0.9938492953342953, 0.0027776371820253634],
    'ph': [3.1894997594997596, 0.14275036415415457],
    'sulphates': [0.4865536315536315, 0.10591379697084427],
    'alcohol': [10.552524450853296, 1.1920626760099189],
    'quality': [5.848244348244348, 0.7707114864014636]
    }

    features_distribution_red = {
    'fixed_acidity': [7.717690058479532, 0.9601701519936446],
    'volatile_acidity': [0.4620029239766082, 0.11894457777866924],
    'citric_acid': [0.2708187134502924, 0.13168322120676834],
    'residual_sugar': [2.442836257309941, 1.2378368096371026],
    'chlorides': [0.07615058479532164, 0.013052405340114065],
    'free_sulfur_dioxide': [17.180555555555557, 9.930354207357635],
    'total_sulfur_dioxide': [51.45029239766082, 33.41704420281052],
    'density': [0.9962276900584796, 0.0015504684741867987],
    'ph': [3.3351023391812866, 0.11341428763051377],
    'sulphates': [0.6233040935672515, 0.10275796199492253],
    'alcohol': [10.407748538011695, 1.0509487679974698],
    'quality': [5.657894736842105, 0.7096360071864578]
    }

    # For the categorical 'type' column, you need to provide the distribution. 
    type_distribution = {'white': 0.858736, 'red': 0.141264}
    new_categorical_data = {'type': np.random.choice(list(type_distribution.keys()), p=list(type_distribution.values()))}
    
    if new_categorical_data == "white":
        # For numerical columns, use the specified mean and std to sample from a Gaussian distribution
        new_numerical_data = {column: np.random.normal(mean, std) for column, (mean, std) in features_distribution_white.items()}
    else:
        # For numerical columns, use the specified mean and std to sample from a Gaussian distribution
        new_numerical_data = {column: np.random.normal(mean, std) for column, (mean, std) in features_distribution_red.items()}
    

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
