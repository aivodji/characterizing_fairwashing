from __future__ import print_function

from functools import partial

# utils
import pickle
import argparse
import os
import pandas as pd 
import numpy as np 
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import load_model

from metrics import ConfusionMatrix, Metric

from collections import Counter




data_dict = {
    'adult_income'      : ('adult_income', 'income'),
    'compas'            : ('compas', 'low_risk'),
    'default_credit'    : ('default_credit', 'good_credit'),
    'marketing'         : ('marketing', 'subscribed')      
}


data_map = {
    'adult_income'      : 'Adult Income',
    'compas'            : 'COMPAS',
    'default_credit'    : 'Default Credit',
    'marketing'         : 'Marketing'      
}



subgroup_dict = {
    'adult_income'      : ('gender_Female', 'gender_Male'),
    'compas'            : ('race_African-American', 'race_Caucasian'),
    'default_credit'    : ('SEX_Female', 'SEX_Male'),
    'marketing'         : ('age_age:30-60', 'age_age:not30-60')      
}
    
def prepare_data(data, rseed):

    dataset, decision = data_dict[data]
    datadir = '../preprocessing/preprocessed/{}/'.format(dataset)    

    #filenames
    train_file      = '{}{}_trainOneHot_{}.csv'.format(datadir, dataset, rseed)
    test_file       = '{}{}_testOneHot_{}.csv'.format(datadir, dataset, rseed)
    sg_file         = '{}{}_attackOneHot_{}.csv'.format(datadir, dataset, rseed)

    # load dataframe
    df_train    = pd.read_csv(train_file)
    df_test     = pd.read_csv(test_file)
    df_sg       = pd.read_csv(sg_file)

    # prepare the data
    scaler = StandardScaler()
    ## training set
    y_train = df_train[decision]
    X_train = df_train.drop(labels=[decision], axis = 1)
    X_train = scaler.fit_transform(X_train)
    ### cast
    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)

    ## test set
    y_test = df_test[decision]
    X_test = df_test.drop(labels=[decision], axis = 1)
    X_test = scaler.fit_transform(X_test)
    ### cast
    X_test = np.asarray(X_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)

     ## sg set
    y_sg = df_sg[decision]
    X_sg = df_sg.drop(labels=[decision], axis = 1)
    X_sg = scaler.fit_transform(X_sg)
    ### cast
    X_sg = np.asarray(X_sg).astype(np.float32)
    y_sg = np.asarray(y_sg).astype(np.float32)

    return X_train, y_train, X_test, y_test, X_sg, y_sg


def prepare_data_as_dataframe(data, rseed):

    dataset, _ = data_dict[data]
    datadir = '../preprocessing/preprocessed/{}/'.format(dataset)    

    #filenames
    train_file      = '{}{}_trainOneHot_{}.csv'.format(datadir, dataset, rseed)
    test_file       = '{}{}_testOneHot_{}.csv'.format(datadir, dataset, rseed)
    sg_file         = '{}{}_attackOneHot_{}.csv'.format(datadir, dataset, rseed)

    # load dataframe
    df_train    = pd.read_csv(train_file)
    df_test     = pd.read_csv(test_file)
    df_sg       = pd.read_csv(sg_file)

    return df_train, df_test, df_sg


def get_metrics(dataset, model_class, rseed):

    # load data as np array
    X_train, y_train, X_test, y_test, X_sg, y_sg = prepare_data(dataset, rseed)
    
    # load data as dataframe
    df_train, df_test, df_sg = prepare_data_as_dataframe(dataset, rseed)

    # load meta data for fairness metrics
    _, decision = data_dict[dataset]
    min_feature, maj_feature = subgroup_dict[dataset]

    # model path
    outdir = '../models/pretrained/{}/'.format(dataset)
    model_path = '{}{}_{}.h5'.format(outdir, model_class, rseed)
   

    def get_predictions(model_class, X_train, y_train, X_test, y_test, X_sg, y_sg):
        predictions_train, predictions_test, predictions_sg = None, None, None
        acc_train, acc_test, acc_sg = None, None, None

        prediction_metrics = {}
        
        if model_class == 'DNN':
            # load model
            mdl = load_model(model_path)

            # get prediction
            #---train
            predictions_train = (mdl.predict(X_train) > 0.5).astype('int32')
            predictions_train = [x[0] for x in predictions_train]
            print(Counter(predictions_train))


            #---test
            predictions_test = (mdl.predict(X_test) > 0.5).astype('int32')
            predictions_test = [x[0] for x in predictions_test]

            #---sg
            predictions_sg = (mdl.predict(X_sg) > 0.5).astype('int32')
            predictions_sg = [x[0] for x in predictions_sg]

            # get accuracy
            acc_train = mdl.evaluate(X_train, y_train)[1]
            acc_test = mdl.evaluate(X_test, y_test)[1]
            acc_sg = mdl.evaluate(X_sg, y_sg)[1]
        
        if model_class in ['RF', 'SVM', 'AdaBoost', 'XgBoost']:
            # load model
            mdl = pickle.load(open(model_path,"rb"))

            # get prediction
            #---train
            predictions_train = mdl.predict(X_train)
            predictions_train = [int(x) for x in predictions_train]

            #---test
            predictions_test = mdl.predict(X_test)
            predictions_test = [int(x) for x in predictions_test]

            #---sg
            predictions_sg = mdl.predict(X_sg) 
            predictions_sg = [int(x) for x in predictions_sg]

            # get accuracy
            acc_train   = accuracy_score(y_train, mdl.predict(X_train))
            acc_test    = accuracy_score(y_test, mdl.predict(X_test))
            acc_sg      = accuracy_score(y_sg, mdl.predict(X_sg))

        #----train
        prediction_metrics['predictions_train'] = predictions_train
        prediction_metrics['acc_Train'] = acc_train

        #----test
        prediction_metrics['predictions_test'] = predictions_test
        prediction_metrics['acc_Test'] = acc_test

        #----sg
        prediction_metrics['predictions_sg'] = predictions_sg
        prediction_metrics['acc_Suing Group'] = acc_sg


        return prediction_metrics

    
    def get_fairness_metrics(df_train, df_test, df_sg, prediction_metrics):
        # output object
        fairness_metrics = {}

        #----train
        df_train['predictions'] = prediction_metrics['predictions_train']
        cm_train = ConfusionMatrix(df_train[min_feature], df_train[maj_feature], df_train['predictions'], df_train[decision])
        cm_minority_train, cm_majority_train = cm_train.get_matrix()
        fm_train = Metric(cm_minority_train, cm_majority_train)


        #----test
        df_test['predictions'] = prediction_metrics['predictions_test']
        cm_test = ConfusionMatrix(df_test[min_feature], df_test[maj_feature], df_test['predictions'], df_test[decision])
        cm_minority_test, cm_majority_test = cm_test.get_matrix()
        fm_test = Metric(cm_minority_test, cm_majority_test)

        #----sg
        df_sg['predictions'] = prediction_metrics['predictions_sg']
        cm_sg = ConfusionMatrix(df_sg[min_feature], df_sg[maj_feature], df_sg['predictions'], df_sg[decision])
        cm_minority_sg, cm_majority_sg = cm_sg.get_matrix()
        fm_sg = Metric(cm_minority_sg, cm_majority_sg)

        fairness_metrics['Train']       = fm_train
        fairness_metrics['Test']        = fm_test
        fairness_metrics['Suing Group'] = fm_sg

        return fairness_metrics

    
    def get_output(dataset, model_class, output_type, prediction_metrics, fairness_metrics):
        res = {}

        # dataset
        res['Dataset']  = data_map[dataset]

        # model class
        res['Model']    = model_class

        # output type
        res['Partition']     = output_type

        # accuracy
        res['Accuracy'] = np.round(prediction_metrics['acc_{}'.format(output_type)], 3)
        
        # fairness
        res['SP']       = np.round(fairness_metrics['{}'.format(output_type)].fairness_metric(1), 3)
        res['PE']       = np.round(fairness_metrics['{}'.format(output_type)].fairness_metric(3), 3)
        res['EOpp']     = np.round(fairness_metrics['{}'.format(output_type)].fairness_metric(4), 3)
        res['EOdds']    = np.round(fairness_metrics['{}'.format(output_type)].fairness_metric(5), 3)

        return res

    prediction_metrics = get_predictions(model_class, X_train, y_train, X_test, y_test, X_sg, y_sg)
    fairness_metrics = get_fairness_metrics(df_train, df_test, df_sg, prediction_metrics)

    output_train    = get_output(dataset, model_class, 'Train', prediction_metrics, fairness_metrics)
    output_test     = get_output(dataset, model_class, 'Test', prediction_metrics, fairness_metrics)
    output_sg       = get_output(dataset, model_class, 'Suing Group', prediction_metrics, fairness_metrics)

    return output_train, output_test, output_sg



if __name__ == '__main__':
    # inputs
    datasets = ['adult_income', 'compas', 'default_credit', 'marketing']
    model_classes = ['AdaBoost', 'DNN', 'RF', 'XgBoost']
    
    df_list = []

    for rseed in range(10):
        row_list = []
        for dataset in datasets:
            for model_class in model_classes:
                output_train, output_test, output_sg = get_metrics(dataset, model_class, rseed)
                row_list.append(output_train)
                row_list.append(output_test)
                row_list.append(output_sg)
        df = pd.DataFrame(row_list)
        df_list.append(df)

    
    
    average_row_list = []
    for index in range(len(df_list[0])):
        average_row = {
            'Dataset'      : df_list[0].iloc[index]['Dataset'],
            'Model'         : df_list[0].iloc[index]['Model'],
            'Partition'      : df_list[0].iloc[index]['Partition'],
            'Accuracy'     : np.round(np.mean([df_list[j].iloc[index]['Accuracy'] for j in range(10)]), 2),
            'SP'     : np.round(np.mean([df_list[j].iloc[index]['SP'] for j in range(10)]), 2),
            'PE'     : np.round(np.mean([df_list[j].iloc[index]['PE'] for j in range(10)]), 2),
            'EOpp'     : np.round(np.mean([df_list[j].iloc[index]['EOpp'] for j in range(10)]), 2),
            'EOdds'     : np.round(np.mean([df_list[j].iloc[index]['EOdds'] for j in range(10)]), 2)
        }
        average_row_list.append(average_row)
    df_average = pd.DataFrame(average_row_list)
    
    
    save_dir = "./results/summary"

    os.makedirs(save_dir, exist_ok=True)

    filename = '{}/summary.csv'.format(save_dir)

    

    df_average.to_csv(filename, encoding='utf-8', index=False)



