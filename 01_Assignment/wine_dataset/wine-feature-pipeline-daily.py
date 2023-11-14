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
    Returns a DataFrame containing one random iris flower
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


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    iris_df = get_random_wine()

    iris_fg = fs.get_feature_group(name="wine",version=1)
    iris_fg.insert(iris_df)

if __name__ == "__main__":
    g()
