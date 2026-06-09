'''
output file contains jsonl file with following format:
{"input", "output", "GT_annotation", "full_response", "weighted_score", "aggregated_score", "comments", "wightened_score_wocomments", "aggregated_score_wocomments", "full_response_wocomments"}

Runs evaluation on output file and generates csv output
'''
import json, os, numpy as np, pandas as pd
from sklearn.metrics import *
from script.evaluation.train_tree import train_tree, infer_tree


def percent_NAcomments(comments):
    '''
    input: comments
    description:
        - Calculates the percentage of NA comments per aspect
    output: percentage of NA comments per aspect'''
    cf = []
    for aspect, checks in comments.items():
        c=0
        for check in checks:
            if check == "na":
                c+=1
            
        # print("% of NA comments", aspect, ":", c/len(checks))
        cf.append(c/len(checks))
    return cf

def calc_matches(resp, resp_wo):
    '''
    input are lists of "yes", "no" responses
    description:
        - Calculates the exact and partial matches between two lists
    output: exact, partial
    '''
    flag = 0
    partial = 0
    for r, rw in zip(resp, resp_wo):
        if r != rw:
            flag=1
        else:
            partial += 1    
    return 1-flag, partial/len(resp)


def evaluation(datas, mp, model, aspects, comments_raw, args, type, checklist, iteration, again, output_filepath, thres_NA=100, num=None):
    '''
    input: datas, mp, model, aspects, comments_raw, args, type, checklist, iteration, output_filepath, thres_NA
    description:
        - Evaluates the model on the datas, trained scores
        if args.save_all:
            - Saves the aggregated responses to output_filepath
            - Calculates the correlation and RMSE between predicted and target annotation per aspect
            - Saves the results to summary_{wt/agg}.csv
    output: avg_error, avg_corr, np.mean(comments), aspects, model, datas
    '''
    save_path = f'output/response/summary_'
    if num!=None:
        save_path = f'output/{args.config}__num={num}/response/summary/'
    comments = percent_NAcomments(comments_raw)
    
    # Remove aspects based on comments >= thres_NA
    if type == "train" and args.resp_comments:
        indices_to_remove = [i for i, val in enumerate(comments) if val >= thres_NA]
        aspect_remove = [mp[i] for i in indices_to_remove]
        for asp in aspect_remove:
            if asp in aspects:
                aspects.remove(asp)
    
    wo = "_wocomments"
    for resp_stored in ["full_response", "full_response"+wo]: 
        if model is None:
            # print("...TRAINING MODEL...", type)
            model = train_tree(datas, resp_stored)
        datas = infer_tree(model, datas, resp_stored)
        
        if args.save_all:
            # print("...saving aggregated responses...")
            with open(output_filepath, 'w', encoding="utf-8") as f: # updating responses from trained tree
                for data in datas:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')

        # correlation and RMSE between predicted and target annotation per aspect
        preds = ["weighted_score", "aggregated_score"]
        files = ["wt", "agg"]
        if "wocomments" in resp_stored:
            preds = [pred+wo for pred in preds] # weighted_score_wocomments, aggregated_score_wocomments
            files = [file+wo for file in files] # wt_wocomments, agg_wocomments

        for pred, file in zip(preds, files):
            predicted = {}
            target = {}
            error = {}
            exact_match_per_aspect = {}
            partial_match_per_aspect = {}
            for key in datas[0]["GT_annotation"].keys(): # aspects
                if key not in predicted:
                    exact_match_per_aspect[key] = []
                    partial_match_per_aspect[key] = []
                exact, partial=[], []
                for data in datas:
                    if args.resp_comments!=2:
                        continue
                    exact_val, partial_val = calc_matches(data['full_response'][key], data['full_response'+wo][key])
                    exact.append(exact_val)
                    partial.append(partial_val)
                exact_match_per_aspect[key]=np.average(exact)
                partial_match_per_aspect[key]=np.average(partial)
            exact_match = np.round(np.mean(list(exact_match_per_aspect.values())), 3)
            partial_match = np.round(np.mean(list(partial_match_per_aspect.values())), 3)
            for data in datas:
                for key in data["GT_annotation"].keys():
                    if key not in predicted:
                        predicted[key] = []
                        target[key] = []
                    predicted[key].append(data[pred][key])
                    target[key].append(data["GT_annotation"][key])
                    if key not in error:
                        error[key] = [] # RMSE: root mean squared error
                    rmse = root_mean_squared_error([data["GT_annotation"][key]], [data[pred][key]])
                    error[key].append(np.round(rmse, 3))

            corr = {}
            for key in predicted.keys():
                if np.std(target[key]) == 0 or np.std(predicted[key])==0: # for if constant values
                    target[key].append(0)
                    predicted[key].append(0)
                
                corr[key] = np.corrcoef(target[key], predicted[key])[0, 1]
            try:
                error_per_aspect = np.round(list(np.mean(list(error.values()), axis=1)), 3)
            except:
                print(error)
                error_per_aspect = np.round(list(np.mean(list(error.values()), axis=0)), 3)
                                            
            avg_error = np.round(np.mean(error_per_aspect   ), 3)
            avg_corr  = np.round(np.mean(list(corr.values())), 3)

            # print(f"{type.upper()}[{file}]- RMSE: ", [np.round(e, 3) for e in error_per_aspect], "CORRELATION: ", [np.round(cor, 3) for cor in corr.values()])

            len_check = {}
            for aspect, check in checklist.items():
                len_check[aspect.lower()]=len(check)

            if args.save_all:
                if os.path.exists(save_path+f'{file}.csv'):
                    result = pd.read_csv(save_path+f'{file}.csv')
                    i = len(result)
                else:
                    result = pd.DataFrame([], columns=['dataset', 'iteration', 'type', 'avg_len_checklist', 'avg_error', 'avg_corr', 'avg_exact_match', 'avg_partial_match'])
                    for aspect in aspects:
                        result[f'error_{aspect}'] = 0
                        result[f'corr_{aspect}'] = 0
                        result[f'comments_{aspect}'] = 0
                        result[f'len_check_{aspect}'] = 0
                        result[f'exact_match_{aspect}'] = 0
                        result[f'partial_match_{aspect}'] = 0
                    i=0
                
                result.loc[i, 'dataset'] = type
                result.loc[i, 'iteration'] = iteration
                result.loc[i, 'type'] = again
                result.loc[i, 'avg_len_checklist'] = np.round(np.mean(list(len_check.values())), 3)
                result.loc[i, 'avg_error'] = avg_error
                result.loc[i, 'avg_corr'] = avg_corr
                result.loc[i, 'avg_exact_match'] = exact_match
                result.loc[i, 'avg_partial_match'] = partial_match
                for aspect in aspects:
                    for key in mp.keys():
                        if mp[key] == aspect:
                            aspect_index = key
                            break
                    result.loc[i, f'error_{aspect}'] = error_per_aspect[aspect_index]
                    result.loc[i, f'corr_{aspect}'] = np.round(corr[aspect.lower()], 3)
                    result.loc[i, f'comments_{aspect}'] = comments[aspect_index]
                    result.loc[i, f'len_check_{aspect}'] = len_check[aspect.lower()]
                    result.loc[i, f'exact_match_{aspect}'] = exact_match_per_aspect[aspect.lower()]
                    result.loc[i, f'partial_match_{aspect}'] = partial_match_per_aspect[aspect.lower()]

                # remove duplicates
                result = result.drop_duplicates()
                result.to_csv(save_path+f'{file}.csv', index=False)
        if args.resp_comments!=2: break

    return aspects, model, datas