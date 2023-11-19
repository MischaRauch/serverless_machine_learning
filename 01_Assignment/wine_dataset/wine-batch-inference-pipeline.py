import os
import modal 

LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn==1.1.1","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests
    from pandas.plotting import table
    import matplotlib.pyplot as plt
    import random
    
 

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=3)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")
    
    feature_view = fs.get_feature_view(name="wine", version=3)
    batch_data = feature_view.get_batch_data()
    
    # change type to binary
    batch_data['type'] = batch_data['type'].replace({'white': 0, 'red': 1})
    y_pred = model.predict(batch_data)
    # offset 3668 & 4812 = 5
    # offset 2568 = 7
    # offset 2284 = 4
    # we have 6,5,7,4,8
    offset = 1
    #while (y_pred[y_pred.size-offset] == 6 or y_pred[y_pred.size-offset] == 5 or y_pred[y_pred.size-offset] == 7 or y_pred[y_pred.size-offset] == 4 or y_pred[y_pred.size-offset] == 8 or y_pred[y_pred.size-offset] == 3):
    #    offset = random.randint(0,y_pred.size)
    #print("OFFSET: ",offset)
    wine = y_pred[y_pred.size-offset]
    print("WINE IS: ",wine)
    print("TYPE: ",type(wine))
    
    wine_df = pd.DataFrame({'predicted': [wine]})
    # Set the index to None
    dfi.export(wine_df,"latest_wine.png")
    

    #img = Image.open(requests.get(flower_url, stream=True).raw)            
    #img.save("./latest_iris.png")
    dataset_api = project.get_dataset_api()    
    dataset_api.upload("./latest_wine.png", "Resources/images", overwrite=True)
   
    wine_fg = fs.get_feature_group(name="wine", version=3)
    df = wine_fg.read() 
    
    label = df.iloc[-offset]["quality"]
    entire_data_row = df.iloc[-offset]
    #label_url = "https://raw.githubusercontent.com/featurestoreorg/serverless-ml-course/main/src/01-module/assets/" + label + ".png"
    #print("Flower actual: " + label)
    #img = Image.open(requests.get(label_url, stream=True).raw)            
    print("LABEL IS: ",label)
    print("ENTIRE ROW: ",entire_data_row)
    print("TYPE OF ROW: ",type(entire_data_row))
    dfi.export(pd.DataFrame(entire_data_row),"actual_wine.png")
    
    dataset_api.upload("./actual_wine.png", "Resources/images", overwrite=True)
    
    monitor_fg = fs.get_or_create_feature_group(name="wine_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Wine Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [wine],
        'label': [label],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])


    df_recent = history_df.tail(4)
    dfi.export(df_recent, './df_recent_wine.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent_wine.png", "Resources/images", overwrite=True)
    
    print("HISTROY: ",history_df)
    predictions = history_df[['prediction']].round().astype(int)
    labels = history_df[['label']].round().astype(int)

    # Only create the confusion matrix when our iris_predictions feature group has examples of all 3 iris flowers
    print("Number of different wine predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() >= 7:
        results = confusion_matrix(labels, predictions)
        print("RSULTS: ",results)
        df_cm = pd.DataFrame(results, ['True 3', 'True 4', 'True 5', 'True 6', 'True 7', 'True 8', 'True 9'],
                             ['Pred 3', 'Pred 4', 'Pred 5', 'Pred 6', 'Pred 7', 'Pred 8', 'Pred 9'])
    
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix_wine.png")
        dataset_api.upload("./confusion_matrix_wine.png", "Resources/images", overwrite=True)
    else:
        print("You need 3 different wine predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 3 different wine predictions") 


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()

