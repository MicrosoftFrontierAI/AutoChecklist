from sklearn import tree
import json, numpy as np


def boolean2binary(input_list):
    '''
    input: list of responses of string type
    description:
        - Converts the list of responses to integer type
    output: integer list of responses
    '''
    if input_list[0].lower() in ['yes', 'no', 'na']: # boolean
        return [1 if item.lower() == 'yes' else 0 for item in input_list]
    else: # ratings
        if type(input_list[0]) is str: 
            temp=[]
            for s in input_list:
                if s.isdigit(): temp.append(float(s))
                else: temp.append(1 if s.lower() == 'yes' else 0)
            return temp
        else:
            temp=[]
            for item in input_list:
                if item.lower() is "na": continue
                try: item=int(item)
                except: item = [int(s) for s in item.split() if s.isdigit()][0]
                temp.append(item)
            return temp

def train_tree(data, resp_stored="full_response", label="GT_annotation"):
    '''
    input: data, label
    description:
        - Trains a DecisionTreeRegressor() model for each aspect
    output: model
    '''
    Xtrain={}
    Ytrain={}
    for d in data:
        for aspect in d[resp_stored].keys():
            if aspect not in Xtrain.keys():
                Xtrain[aspect.lower()] = []
                Ytrain[aspect.lower()] = []
            temp = boolean2binary(d[resp_stored][aspect.lower()])
            Xtrain[aspect.lower()].append(temp)
            Ytrain[aspect.lower()].append(d[label][aspect.lower()])

    model = {}
    for aspect in Xtrain.keys():
        clf = tree.DecisionTreeRegressor()
        clf = clf.fit(Xtrain[aspect.lower()], Ytrain[aspect.lower()])
        model[aspect.lower()] = clf
    return model

def infer_tree(model, data, resp_stored="full_response", label="GT_annotation"): 
    '''
    input: model, data, label
    description:
        - Infer the model on the data
    output: data
    '''
    Xtest={}
    Ytest={}
    for d in data:
        for aspect in d[resp_stored].keys():
            if aspect not in Xtest.keys():
                Xtest[aspect.lower()] = []
                Ytest[aspect.lower()] = []
            temp = boolean2binary(d[resp_stored][aspect.lower()])
            Xtest[aspect.lower()].append(temp)
            Ytest[aspect.lower()].append(d[label][aspect.lower()])
    
    Ypred = {}
    for aspect in Xtest.keys():
        Ypred[aspect.lower()] = model[aspect.lower()].predict(Xtest[aspect.lower()])
        # print("ERROR IN INFER:", np.sqrt(np.sum((Ypred[aspect.lower()]-Ytest[aspect.lower()])**2)))

    respond_op = "weighted_score"
    respond_op += "_wocomments" if "comments" in resp_stored else ""
    for i,d in enumerate(data):
        for aspect in d[resp_stored].keys():
            d[respond_op][aspect.lower()] = Ypred[aspect.lower()][i]
    return data

def feature_importance(model, checklists, saving_path, resp_path=None):
    '''
    input: model, checklists, saving_path, resp_path
        model: trained DecisionTreeRegressor(), importance lie in [0, 1]
    description:
        - Calculate the feature importance of the model
        1. individual: correlation based on individual yes/no vs gt annotations
            - [-1, 1]: negatively to positively correlated
            
            # NOT CONSIDERED DURING GT ANNOTATION:
            - [2, 3]: all response was [0, 1] - for different annotations
            - [4, 9]: different responses - for all annotations of [0, 5]
        2. normalized: normalized individual correlation
            - lie between [0, 1] similar  to model importance
            - sum = 1
            - keeping >1.5 labels intact
    output: None
    '''
    updated_checklists = {}
    for key, checklist in checklists.items():
        updated_checklists[key]={}
        if resp_path:
            with open(resp_path, 'r', encoding='utf-8') as f:
                responses = [json.loads(line) for line in f]

            feature_importance_indi=[]
            normalized_feature_importance_indi=[]
            temp=[]
            for i in range(len(checklist)):
                x=[]
                y=[]
                for response in responses:
                    x.append(response['full_response'][key][i])
                    y.append(response['GT_annotation'][key])
                x = boolean2binary(x)
                if np.std(x) == 0:
                    feature_importance_indi.append(2.0+x[0]) # [2, 3]
                elif np.std(y) == 0: 
                    feature_importance_indi.append(4.0+y[0]) # [4, 9]
                else:
                    feature_importance_indi.append(np.corrcoef(x, y)[0, 1]) # [-1, 1]
                    temp.append(np.abs(np.corrcoef(x, y)[0, 1]))
            
            # range: [0, 1]; sum = 1
            for i in range(len(feature_importance_indi)):
                if feature_importance_indi[i]>1.5: # one of labeled cases
                    normalized_feature_importance_indi.append(feature_importance_indi[i])
                else:
                    normalized_feature_importance_indi.append(np.abs(feature_importance_indi[i])/sum(temp))
        else:
            feature_importance_indi = [None]*len(checklist)
            normalized_feature_importance_indi = [None]*len(checklist)
        feature_importance_model = model[key].feature_importances_
        for i, checks in enumerate(checklist):            
            updated_checklists[key][checks] = {"model_imp": np.round(feature_importance_model[i],2), "individual_corr": np.round(feature_importance_indi[i],2), "normalized_imp": np.round(normalized_feature_importance_indi[i],2)}
    
    with open(saving_path, 'w', encoding='utf-8') as f:
        json.dump(updated_checklists, f, ensure_ascii=False, indent=4)
    return 