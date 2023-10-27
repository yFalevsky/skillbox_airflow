import dill
import pandas as pd
import json
import os
from datetime import datetime
import sklearn

path = os.environ.get('PROJECT_PATH', '.')


def predict():

    latest_model = sorted(os.listdir(f'{path}/data1/models'))

    with open(fr'{path}/data1/models/{latest_model[-1]}', 'rb') as f:
        model = dill.load(f)

    test_cars = os.listdir(rf'{path}/data1/test')

    preds = pd.DataFrame(columns=['car_id', 'pred'])

    for elem in test_cars:
        with open(fr'{path}/data1/test/{elem}', 'rb') as file:
            car = json.load(file)
            df = pd.DataFrame(car, index = [0])
            y = model.predict(df)
            dict_pred = {'car_id':df['id'].values[0], 'pred':[0]}
            df2 = pd.DataFrame([dict_pred])
            preds = pd.concat([df2, preds], ignore_index = True)

    now = datetime.now().strftime('%Y%m%d%H%M')
    preds.to_csv(fr'{path}/data1/predictions/{now}.csv', index = False)



if __name__ == '__main__':
    predict()
