'''
FOR PURPOSE OF MULTIPLE EXPERIMENTS:

- generates/continues from the manual seed_checklist provided
- while no stoping condition (max iterations reached/no comments/checklist not updated) encountered:
    - generate (responses+comments)
    - evaluate agnst GT annotation (infer)
    - updating_checklists
    - prune_checklists (optional)
'''
from argparse import ArgumentParser
import json, os, random, warnings, ast
import hashlib, pandas as pd, numpy as np

from script.evaluation.gen import gen
from script.evaluation.eval import evaluation
from script.evaluation.train_tree import feature_importance
from script.updating.updating_checks import updating_checks
from script.updating.pruning_checks import pruning_checks
from script.updating.seed_checks import seed_checks
from script.prompting import get_response

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default="instruct_excel", help="Task to perform evaluation on, [summary, translation, instruct_excel, formual_explaination]")
    parser.add_argument("--dataset_path", type=str, default="instruct_excel_demo.jsonl", help="Path to file containing the dataset in data folder")
    
    ### Preferably do not change
    parser.add_argument("--kfold", type=bool, default=True, help="Flag to note if dataset is to be divided into 5fold cross validation or not")
    parser.add_argument("--aspects", type=str, default='[]', help="List of aspects over which checklist is to be built; Example: ['accuracy', 'overall', 'fluency', '[]', ...]. By default-None, assumes the aspects present within the GT_annotation parameter of dataset. '[]'-signifies no aspect to be used.")
    parser.add_argument("--train_test_size", type=str, default="[40, 10]", help="Total number of train and test data, [Train dataset: checklist updates on comments of these data, Test dataset: only infereneces]")
    parser.add_argument("--resp_comments", type=int, default=2, help="Flags if response should be generated without the comments (0) or with (1) or with both cases are to be saved (2)")
    parser.add_argument("--stoping_iteration", type=int, default=5, help="Max number of iterations to run the script before stopping")
    parser.add_argument("--stoping_precentage", type=int, default=100, help="Stoping condition based on percentage of NA comments in the checklist for each aspect")
    parser.add_argument("--llm", type=str, default="gpt-4o-chat-completions", help="Which LLM (chat/completion) endpoint from your LLM provider is to be used for pipeline. A model name containing 'chat' routes to the chat endpoint; examples: gpt-4o-chat-completions, gpt-4-1106")
    parser.add_argument("--metachecklist", type=str, default="helper/metachecklist.txt", help="File path to txt file with list of metachecklist criteria that each checklist should follow, If do not want to se metachecklist leave the parameter-''(as empty string), metachecklist is additional criteria that needs to be followed by the checklist")

    ### Can be altered based on the task
    parser.add_argument("--custom_checklist", type=str, default="", help="Path to manually curated seed checklist to set as starting point")
    parser.add_argument("--calling_prune", type=int, default=3, help="Flags if we are to use the pruning prompt after each aggregated checklist update iteratively (3), twice at the end (2), only once at the end (1), or not to prune at all (0)")
    
    parser.add_argument("--resp_type", type=int, default=0, help="Flag to note if response is of rating [1-5] (mentioned-1, not-mentioned-2) or boolean(0) [yes/no] type")
    parser.add_argument("--prune_version", type=int, default=2, help="Version of pruning prompt that is to be used, [Version 1-4step instruction of sequentially arranging, removing similar, clustering and combining. Version 2-2step instruction of latter 2 steps. Version 3-no steps just instructed to remove redundant/dupplicate and combine similar.]")
    parser.add_argument("--update_version", type=int, default=1, help="Version of prompt to update checklist that is to be used, [Version 1-produce modified checklist, Version 2-produce only next appending checks generated from comments.]")
    parser.add_argument("--seed_w_sample", type=int, default=0, help="Flags if we are to generate the seed checklist on showcasing 3 sample inputs or not (0), wherein if selecting samples (1) stands for diverse samples, (2) for random samples, (3) high scoring samples, (4) low scoring samples")
    
    ### Debugging and saving
    parser.add_argument("--save_all", type=bool, default=True, help="Debug mode to save all intermediate steps")
    parser.add_argument("--eval", type=int, default=1, help="Reports evaluation metrics over each iteration on aggregating and training regression tree model on train dataset")

    return parser.parse_args()

def generate_evaluate_response(num, cur_aspects, mapped_aspects, train_data, test_data, args, checklists, i, again=0):
    '''
    input: train_data, test_data, args, checklists, i-iteration
    description:
        - generate response on test and train data
        - evaluate the scores
        - update the aspects based on stopping condition
    output: responses, cur_aspects, stopping_condition
    '''
    stopping_condition=False
    # generate response on test and train data
    responses = {}
    for dataset, data_type in zip([test_data, train_data], ["test", "train"]):
        if dataset:
            cleaned_comments, comments, response = generate_response(num, dataset, args, checklists, i, data_type, again)
            responses[data_type] = response
    c = 0
    for aspect in cleaned_comments.keys():
        c+=len(cleaned_comments[aspect.lower()])
    if args.resp_comments and c==0:
        stopping_condition=True 
        print("Stoping: No comments available on train data--")

    if args.eval:
        # generate aggregated and train regression model scores, to evaluate the scores
        model=None
        for dataset, data_type in zip([train_data, test_data], ["train", "test"]):
            if dataset:
                [modified_aspects, model, response] = evaluation(datas = responses[data_type], mp = mapped_aspects, model = model, aspects = cur_aspects.copy(), comments_raw = comments, args = args, type = data_type, checklist = checklists, iteration = i, again=again, output_filepath = f"output/{args.config}__num={num}/response/{data_type}_{i}_{again}.jsonl", thres_NA = args.stoping_precentage, num=num)
                responses[data_type] = response
            
            if data_type=="train": 
                cur_aspects = modified_aspects
                if args.save_all:
                    feature_importance(model, checklists, f"output/{args.config}__num={num}/checklist/feature_importance/checklist_{i}_{again}.json", resp_path = f"output/{args.config}__num={num}/response/{data_type}_{i}_{again}.jsonl")
        
    else:
        if args.resp_comments:
            for aspect in cleaned_comments:
                if len(cleaned_comments[aspect.lower()])*100==args.stoping_precentage*len(comments[aspect.lower()]): 
                    # if comments available are less than stopping percent, ignore them for the aspect
                    cur_aspects.remove(aspect.lower())
    
    return responses, cur_aspects, cleaned_comments, stopping_condition

def pruning_checklist(num, i, again, aspects, args, checklists, list_prev):
    '''
    input: i-iteration, again-iteration of pruning, aspects, args, checklists, list_prev
    description:
        - prune the checklist
        - check if the pruned checklist is already present in the list of previous checklists
    output: pruned_checklists
    '''
    again+=1
    stopping_condition=False
    if os.path.exists(f"output/{args.config}__num={num}/checklist/checklist_{i}_{again}.json"):
        with open(f"output/{args.config}__num={num}/checklist/checklist_{i}_{again}.json", 'r') as f:
            pruned_checklists = json.load(f)
    else:
        pruned_checklists = pruning_checks(aspects, args, f"output/{args.config}__num={num}/checklist/checklist_{i}_{again}.json", checklists)

    if is_in(pruned_checklists, list_prev):
        # os.remove(f"output/{args.config}__num={num}/checklist/checklist_{i}_{again+1}.json")
        stopping_condition=True 
        print('Pruning did not update anything')

    return pruned_checklists, stopping_condition, again

def generate_response(num, data, args, checklists, i, data_type, again=0):
    '''
    input: data, args, checklists, i-iteration, data_type-train/test
    description:
        - generate response on given dataset
        - post process comments: remove duplicates, and N/A values
        - save the comments to file
    output: cleaned_comments, data
    '''
    # generate response on given dataset
    file = f"output/{args.config}__num={num}/response/{data_type}_{i}_{again}.jsonl"
    if not os.path.exists(file):
        comments, response = gen(data, args, file, checklists)
    else:
        comments = load_saved_comments(args, num, i, data_type)
        with open(file, 'r', encoding='utf-8') as f:
            response = [json.loads(line) for line in f]
    print(f"GENERATED Responses on {data_type} data")

    # post process comments: remove duplicates, and N/A values
    cleaned_comments = {k: [a for a in v if a!="na"] for k, v in comments.items()}
    for k, v in cleaned_comments.items():
        cleaned_comments[k] = list(set(v))

    # save the comments to file
    if args.save_all:
        with open(f"output/{args.config}__num={num}/metadata/comments/{data_type}_{i}_{again}.json", 'w') as f:
            json.dump(cleaned_comments, f)
    return cleaned_comments, comments, response

def load_saved_comments(args, num, i, data_type):
    '''
    input: i-iteration
    description:
        - load the saved comments from the output file
    output: comments
    '''
    comments={}
    again=3
    while not os.path.exists(f"output/{args.config}__num={num}/response/{data_type}_{i}_{again}.jsonl"):
        again-=1
    # load the saved comments from the train data output file
    with open(f"output/{args.config}__num={num}/response/{data_type}_{i}_{again}.jsonl", 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    for d in data:
        for key, value in d['comments'].items():
            if key not in comments:
                comments[key] = []
            comments[key].append(value)
    return comments

def is_in(checklists, list_prev):
    '''
    input: 
        checklists-current checklist, 
        list_prev-previouly generated checklists
    description:
        - checks if the current checklist is already present in the list of previous checklists
    output: True/False    
    '''
    for listed in list_prev:
        f=0
        for aspect, checks in listed.items():
            criteria = checklists[aspect.lower()]
            if set(criteria)!=set(checks):
                f=1
                break
        if f:
            continue
        else:
            return True
    return False

def wrapper_args():
    '''
    description:
        - wrapper function to assert and initialize the parameters and folders
        - read the dataset
        - initialize the output folders and train, test datasets
        - load previously saved task and aspect descriptions
    output: args, description, train_data, test_data
    '''
    # defining helper functions
    def add_desc(args, samples, description, type):
        '''
        input: 
            args-arguments,
            samples-3 random samples from the dataset,
            description-task and aspect descriptions,
            type-task/aspect
        description:
            - prompts LLM to generate task/aspect description
            - updates in utils description
        output: description with updated task/aspect descriptions
        '''
        with open(f"helper/prompts/{type}.txt", 'r') as f:
            prompt = f.read()  
        prompt = prompt.replace("{task}", args.task).replace("task_desc", description[args.task]['description']).replace("{samples}", samples)
        hist=[]
        
        if type=="task":
            desc, _, _ = get_response([], prompt) 
            # llm model is default gpt-4o to keep all the responses consistent
            print("Task description added:\n", args.task, ':', desc[0])
            description[args.task.lower()] = {"description": desc[0], "aspects": {}}
        
        if type=="aspect":
            print("For the task of:", args.task)
            for aspect in args.aspects:
                if aspect=='NA':
                    description[args.task.lower()]['aspects'][aspect.lower()] = ""

                desc, _, hist = get_response(hist, prompt.replace("{aspect}", aspect)) 
                # llm model is default gpt-4o to keep all the responses consistent
                print("Aspect description added:\n", aspect, ':', desc[0])
                description[args.task.lower()]['aspects'][aspect.lower()] = desc[0]
        
        return description

    def asserting_inputs(args):
        '''
        input: args
        description:
            - assert the parameters are of correct type
        '''
        # asserting parameters are of correct type
        assert os.path.exists(args.dataset_path), "Dataset file not found in data folder"
        if args.custom_checklist:
            assert os.path.exists(args.custom_checklist), "Seed checklist file not found"
        if args.metachecklist and args.metachecklist.strip()!='':
            if not os.path.exists(args.metachecklist):
                warnings.warn("Metachecklist file not found. thus setting as empty string")
                args.metachecklist = None
        assert len(args.train_test_size)==2, "Train and test dataset size is not 2, but of length "+str(len(args.train_test_size))
        assert args.stoping_precentage<=100 and args.stoping_precentage>=0, "Stopping percentage can only be within the values of less than 100"
        assert args.resp_type in [0, 1, 2], "Response type can only be within the values of 0, 1 or 2"
        assert args.prune_version in [1, 2, 3], "Prune version can only be within the values of 1, 2 or 3"
        assert args.update_version in [1, 2], "Update version can only be within the values of 1 or 2"
        assert args.calling_prune in [0, 1, 2, 3], "Calling prune can only be within the values of 0, 1, 2 or 3"
        assert args.seed_w_sample in [0, 1, 2, 3, 4], "Seed with sample can only be within the values of 0, 1, 2, 3 or 4"
        assert args.resp_comments in [0, 1, 2], "Response comments can only be within the values of 0, 1 or 2"
        
        # read the complete dataset
        with open(args.dataset_path, 'r', encoding='utf-8') as f: 
            data = [json.loads(line) for line in f]
        if args.aspects == None:
            args.aspects = list(data[0]['GT_annotation'].keys())
        if args.aspects==[]: args.aspects = ['na']

        is_present = set(args.aspects).issubset(set(data[0]['GT_annotation'].keys()))
        assert is_present or not args.eval, "Aspect not found in dataset, can not compare evaluation, Format to include desired aspect in GT_annotation, or set eval to False"
        assert args.train_test_size[0]+args.train_test_size[1]<=len(data), "Sum of train and test dataset size should not be more than length of dataset: "+str(len(data))
        if args.eval:
            warnings.warn("Evaluation is set to True, make sure the GT_annotation is manually curated and not generated randomly")
        return 

    def format_samples(samples):
        '''
        input: samples-3 random samples from the dataset
        description:
            - format the samples to display in prompt
        output: response
        '''
        response=""
        for i,s in enumerate(samples):
            response+=f"SAMPLE-{i}:\n"
            response+=f"Input: {s['input']}\n"
            if s['output']!="":
                response+=f"Output: {s['output']}\n"
            response+="\n"

        return response

    def obtain_test_train_files(args, data):
        '''
        inupt: args, dataset
        description: 
            - obtain the test and train files from the dataset
            - randomly sample the dataset and save it to a file(if args.save_all)
        output: test_file, train_file
        '''
        # randomly sample train and test data and save it to a file
        random.seed(42)  # for reproducibility
        
        if args.kfold:
            random.shuffle(data) 
            # in ratio of 4:1
            temp={}
            for i in range(5):
                temp[i] = data[int(len(data)*i/5):int(len(data)*(i+1)/5)]
            
            trains = {}
            tests = {}
            for i in range(5):
                test_data = temp[i]
                train_data = []
                for j in range(5):
                    if j!=i:
                        train_data+=temp[j]
                trains[i] = train_data
                tests[i] = test_data
            return trains, tests

        train_data = random.sample(data, min(args.train_dataset_size, len(data)))
        test_data = random.sample(data, min(args.test_dataset_size, len(data)))

        # save into data folder
        if args.save_all:
            if train_data and not os.path.exists(args.train_dataset_path+'_train.jsonl'):
                with open(args.train_dataset_path+'_train.jsonl', 'w', encoding='utf-8') as f:
                    for d in train_data:
                        f.write(json.dumps(d, ensure_ascii=False)+'\n')

            if test_data and not os.path.exists(args.train_dataset_path+'_test.jsonl'):
                with open(args.train_dataset_path+'_test.jsonl', 'w', encoding='utf-8') as f:
                    for d in test_data:
                        f.write(json.dumps(d, ensure_ascii=False)+'\n')
        
        return train_data, test_data
    
    def gen_all_folders(args):
        if not os.path.exists(f"output"):
            os.makedirs(f"output")

        if args.kfold:
            nums=[0, 1, 2, 3, 4]
        else:
            nums=[-1]

        for num in nums:
            if not os.path.exists(f"output/{args.config}__num={num}"):
                os.makedirs(f"output/{args.config}__num={num}")
            with open(f"output/{args.config}__num={num}/configs.json", 'w', encoding='utf-8') as f:
                    json.dump(args.__dict__, f, indent=4, ensure_ascii=False)

        if args.save_all:
            folder = ['checklist', 'response', 'metadata']
            for num in nums:
                for f in folder:
                    if not os.path.exists(f"output/{args.config}__num={num}/{f}"):
                        os.makedirs(f"output/{args.config}__num={num}/{f}")
        
            for num in nums:
                for m, f in zip(['checklist', 'response', 'metadata', 'metadata'], ['feature_importance', 'summary', 'comments', 'prune']):
                    if not os.path.exists(f"output/{args.config}__num={num}/{m}/{f}"):
                        os.makedirs(f"output/{args.config}__num={num}/{m}/{f}")
        
        return 
    
    def get_config(args):
        config = ""
        for k, v in args.__dict__.items():
            config+=f"{k}={v}__"
        while config[-1]=="_":
            config = config[:-1]
        #compress
        config_hash = hashlib.md5(config.encode()).hexdigest()
        return config_hash


    args = get_args()
    args.train_test_size = ast.literal_eval(args.train_test_size)
    if args.aspects!=None:
        args.aspects = ast.literal_eval(args.aspects)
    args.dataset_path = os.path.join("data", args.dataset_path)
    asserting_inputs(args)
    
    with open(args.dataset_path, 'r', encoding='utf-8') as f: 
        data = [json.loads(line) for line in f]

    args.train_dataset_path = args.dataset_path.replace(".jsonl", f"_{str(args.train_test_size)}")
    args.train_dataset_size, args.test_dataset_size = args.train_test_size
    
    # initialize output folders and train, test datasets    
    train_data, test_data = obtain_test_train_files(args, data)
    
    args.config = get_config(args)
    # updates the config directory
    with open(f"helper/configs.json",'r', encoding='utf-8') as f:
        configs = json.load(f)
    if args.config not in configs:
        configs[args.config] = args.__dict__
    with open(f"helper/configs.json",'w', encoding='utf-8') as f:
        json.dump(configs, f, indent=4, ensure_ascii=False)

    gen_all_folders(args)

    # load previously saved task and aspect descriptions
    with open("helper/utils.json", 'r') as f:
        description = json.load(f)

    random.seed(42)  # for reproducibility
    samples = random.sample(data, min(3, len(data))) # sample 3 random data
    samples = format_samples(samples)

    # check if task and aspects is already present, if not then add it
    if args.task.lower() not in description.keys():
        print("Encountered a new task, not found in utils")
        # prompts LLM to generate task/aspect description, updates in utils
        description = add_desc(args, samples, description, "task")
    
    if description[args.task]['aspects']=={} or not set([a.lower() for a in args.aspects]).issubset(set([a.lower() for a in description[args.task]['aspects'].keys()])):
        print("Encountered new aspects, not found in utils")
        # prompts LLM to generate task/aspect description, updates in utils
        description = add_desc(args, samples, description, "aspect")
    
    # save the updated description
    with open("helper/utils.json", 'w', encoding='utf-8') as f:
        json.dump(description, f, indent=4, ensure_ascii=False)

    args.description = description  # save the description in args
    return args, train_data, test_data


def wrapped_gen_eval(args, train_data, test_data, num):
    again=0 # updated checklist
    i=0 # iteration
    cur_aspects = args.aspects.copy()
    
    if args.custom_checklist:
        print("Starting from custom seed checklist")
        with open(args.custom_checklist, 'r', encoding="utf-8") as f:
            checklists = json.load(f)
    else:
        checklists = seed_checks(args, f"output/{args.config}__num={num}/checklist/checklist_{i}_{again}.json")
    list_prev = [checklists.copy()] # maintaining list of previous checklists

    stopping_condition = False
    mapped_aspects = {}
    for i, aspect in enumerate(args.aspects):
        mapped_aspects[i] = aspect

    while not stopping_condition:
        print(f"\nITERATION: {i}")
        
        # generate and evaluate responses on train and test data
        responses, cur_aspects, cleaned_comments, stopping_condition = generate_evaluate_response(num, cur_aspects, mapped_aspects, train_data, test_data, args, checklists, i, again)
        
        if i>=args.stoping_iteration: 
            stopping_condition=True
            print("Stoping: Max iterations reached---")
        if stopping_condition:
            break

        if cur_aspects==[]: 
            stopping_condition=True
            print("Stoping: All aspects saturated---")

        i+=1
        # updating the checklist
        checklists = updating_checks(cur_aspects, args, cleaned_comments, f"output/{args.config}__num={num}/checklist/checklist_{i}_{again}.json", checklists)
        
        if is_in(checklists, list_prev): 
            stopping_condition=True
            print("Stoping: Checking not updated---")
        else: 
            list_prev.append(checklists.copy())
        
        if args.calling_prune==3: # prune iteratively
            print("PRUNING ITERATIVELY... at", i, 1)
            checklists, stopping_condition, again = pruning_checklist(num, i, again, cur_aspects, args, checklists, list_prev)
            if again>1: again=1
        if stopping_condition:
            break

    # after reaching stopping condition, prune final checklist and run responses again 
    print()
    if args.calling_prune in [1, 2]: # prune only once at the end
        print("PRUNING ONCE... at", i, again)
        checklists, stopping_condition, again  = pruning_checklist(num, i, again, cur_aspects, args, checklists, list_prev)
        if not stopping_condition:
            # generate and evaluate responses on train and test data
            responses, cur_aspects, cleaned_comments, stopping_condition = generate_evaluate_response(num, cur_aspects, mapped_aspects, train_data, test_data, args, checklists, i, again)

            if args.calling_prune==2:
                print()
                print("PRUNING TWICE... at", i, again)
                checklists, stopping_condition, again = pruning_checklist(num, i, again, cur_aspects, args, checklists, list_prev)
            if not stopping_condition:
                # generate and evaluate responses on train and test data
                responses, cur_aspects, cleaned_comments, stopping_condition = generate_evaluate_response(num, cur_aspects, mapped_aspects, train_data, test_data, args, checklists, i, again)


    # # after reaching stopping condition, save the final checklist and responses
    print()
    with open(f"output/{args.config}__num={num}/checklist.json", 'w', encoding="utf-8") as f:
        json.dump(checklists, f, indent=4, ensure_ascii=False)
    for data_type, resp in responses.items():
        with open(f"output/{args.config}__num={num}/response_{data_type}.jsonl", 'w', encoding="utf-8") as f:
            for r in resp:
                f.write(json.dumps(r, ensure_ascii=False)+'\n')
    # print("Final checklist and responses saved in output folder")
    return

def compile_results(args):
    '''
    extracts the files from eac of num of kfold and compiles the results (response/summary) into a single file
    '''
    if not os.path.exists(f"output/{args.config}"):
        os.makedirs(f"output/{args.config}")

    nums=[0, 1, 2, 3, 4]
    df = pd.DataFrame([], columns=['file','nums','dataset','iteration','type','avg_len_checklist','avg_error','avg_corr', 'avg_exact_match', 'avg_partial_match'])
    for num in nums:
        folder = f"output/{args.config}__num={num}/response/summary/"
        for files in os.listdir(folder):
            file = os.path.join(folder, files)
            df_file = pd.read_csv(file)
            for i in range(len(df_file)):
                df.loc[len(df), :] = {"file": files.split('.')[0], "nums": num, "dataset": df_file.loc[i, 'dataset'], "iteration": df_file.loc[i, 'iteration'], "type": df_file.loc[i, 'type'], "avg_len_checklist": df_file.loc[i, 'avg_len_checklist'], "avg_error": df_file.loc[i, 'avg_error'], "avg_corr": df_file.loc[i, 'avg_corr'], "avg_exact_match": df_file.loc[i, 'avg_exact_match'], "avg_partial_match": df_file.loc[i, 'avg_partial_match']}
    df = df.drop_duplicates()
    df.to_csv(f"output/{args.config}/individual_summary.csv", index=False)
    
    result = df.groupby(['dataset', 'iteration', 'type']).agg({
        'nums': lambda x: set(x),
        'avg_len_checklist': 'mean',
        'avg_error': 'mean',
        'avg_corr': 'mean',
        'avg_exact_match': 'mean',
        'avg_partial_match': 'mean'
    }).reset_index()
    for col in ['avg_len_checklist', 'avg_error', 'avg_corr', 'avg_exact_match', 'avg_partial_match']:
        result[col] = result[col].apply(lambda x: round(x, 4))
    result.to_csv(f"output/{args.config}/summary.csv", index=False)
    
    compiled = pd.DataFrame([], columns=['iteration', 'type', 'nums', 'avg_exact_match', 'avg_partial_match'])
    temp={}
    nums={}
    for col in ['avg_exact_match', 'avg_partial_match']:
        temp[col]={}
        for i in range(args.stoping_iteration):
            iteration = result.loc[result['iteration']==i, ['dataset', col]]
            nums[i] = result.loc[result['iteration']==i, ['type', 'nums']]
            train = iteration.loc[iteration['dataset']=='train']
            test = iteration.loc[iteration['dataset']=='test']
            # temp[col][i] = np.round((train*args.train_dataset_size + test*args.test_dataset_size)/(args.train_dataset_size+args.test_dataset_size), 3)
            temp[col][i] = np.round((train.iloc[0,1]*4 + test.iloc[0,1])/5, 3)
    for _, v in temp.items():
        for i in v.keys():
            compiled.loc[len(compiled), :] = {"iteration": i, "type": nums[i].iloc[0, 0], "nums": nums[i].iloc[0,1], "avg_exact_match": temp['avg_exact_match'][i], "avg_partial_match": temp['avg_partial_match'][i]}
        break
    compiled.loc[len(compiled), :] = {"iteration": "avg", "type": "avg", "nums": "avg", "avg_exact_match": np.round(compiled['avg_exact_match'].mean(), 3), "avg_partial_match": np.round(compiled['avg_partial_match'].mean(), 3)}
    compiled.to_csv(f"output/{args.config}/total_summary.csv", index=False)
    return


def main():
    args, train_data, test_data = wrapper_args()
    if args.kfold:
        for num in range(5):
            print(f"STARTING... TASK: {args.task}\nfor *{num}th* fold")
            wrapped_gen_eval(args, train_data[num], test_data[num], num)
    else:
        print("STARTING... TASK:", args.task)  
        wrapped_gen_eval(args, train_data, test_data, -1)
    
    if args.save_all and args.eval and args.kfold: compile_results(args)
    print("Compiled results, STOPPING...")
    return


if __name__ == "__main__":
    # try:
        main()
    # except Exception as err:
    #     import pdb
    #     pdb.post_mortem()