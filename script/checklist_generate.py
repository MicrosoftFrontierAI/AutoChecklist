'''
FOR INFERENCING IN EVALUATION TOOLS OVER ANY NEW BENCHMARK:

- generates/continues from the manual seed_checklist provided
- while no stoping condition (max iterations reached/no comments/checklist not updated) encountered:
    - generate (responses+comments)
    - updating_checklists (appends on comments)
    - prune_checklists (optimize)
'''
from argparse import ArgumentParser
import json, os, random, warnings, ast

from script.evaluation.gen import gen
from script.updating.updating_checks import updating_checks
from script.updating.pruning_checks import pruning_checks
from script.updating.seed_checks import seed_checks
from script.prompting import get_response

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default="translation", help="Task to perform evaluation on, [summarization, translation, instruct_excel, formual_explaination]")
    parser.add_argument("--dataset_path", type=str, default="example_translation.jsonl", help="Path to file containing the dataset in data folder")
    
    parser.add_argument("--aspects", type=str, default="[]", help="List of aspects over which checklist is to be built; Example: ['accuracy', 'overall', 'fluency', [], ...]. By default-None, assumes the aspects present within the GT_annotation parameter of dataset. []-signifies no aspect to be used.")
    parser.add_argument("--custom_checklist", type=str, default="", help="Path to manually curated seed checklist to set as starting point")
    parser.add_argument("--resp_type", type=int, default=0, help="Flag to note if response is of rating [1-5] (mentioned-1, not-mentioned-2) or boolean(0) [yes/no] type")
    
    parser.add_argument("--stoping_iteration", type=int, default=5, help="Max number of iterations to run the script before stopping")
    parser.add_argument("--llm", type=str, help="Which LLM (chat/completion) endpoint from your LLM provider is to be used for pipeline. A model name containing 'chat' routes to the chat endpoint")
    
    parser.add_argument("--metachecklist", type=str, default="helper/metachecklist.txt", help="File path to txt file with list of metachecklist criteria that each checklist should follow, If do not want to se metachecklist leave the parameter-''(as empty string), metachecklist is additional criteria that needs to be followed by the checklist")
    parser.add_argument("--calling_prune", type=int, default=1, help="Flags if we are to use the pruning prompt after each aggregated checklist update iteratively (3), twice at the end (2), only once at the end (1), or not to prune at all (0)")
    parser.add_argument("--prune_version", type=int, default=3, help="Version of pruning prompt that is to be used, [Version 1-4step instruction of sequentially arranging, removing similar, clustering and combining. Version 2-2step instruction of latter 2 steps. Version 3-no steps just instructed to remove redundant/dupplicate and combine similar.]")
    parser.add_argument("--update_version", type=int, default=1, help="Version of prompt to update checklist that is to be used, [Version 1-produce modified checklist, Version 2-produce only next appending checks generated from comments.]")
    parser.add_argument("--seed_w_sample", type=int, default=0, help="Flags if we are to generate the seed checklist on showcasing 3 sample inputs or not (0), wherein if selecting samples (1) stands for diverse samples, (2) for random samples, (3) high scoring samples, (4) low scoring samples")
    
    parser.add_argument("--save_all", type=bool, default=False, help="Debug mode to save all intermediate steps")

    return parser.parse_args()

def generate_evaluate_response(cur_aspects, mapped_aspects, train_data, test_data, args, checklists, i, again=0):
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
            cleaned_comments, comments, response = generate_response(dataset, args, checklists, i, data_type, again)
            responses[data_type] = response
    c = 0
    for aspect in cleaned_comments.keys():
        c+=len(cleaned_comments[aspect.lower()])
    if args.resp_comments and c==0:
        stopping_condition=True 
        print("Stoping: No comments available--")

    if args.resp_comments:
        for aspect in cleaned_comments:
            if len(cleaned_comments[aspect.lower()])*100==args.stoping_precentage*len(comments[aspect.lower()]): 
                # if comments available are less than stopping percent, ignore them for the aspect
                cur_aspects.remove(aspect.lower())
    
    return responses, cur_aspects, cleaned_comments, stopping_condition

def pruning_checklist(i, again, aspects, args, checklists, list_prev):
    '''
    input: i-iteration, again-iteration of pruning, aspects, args, checklists, list_prev
    description:
        - prune the checklist
        - check if the pruned checklist is already present in the list of previous checklists
    output: pruned_checklists
    '''
    again+=1
    stopping_condition=False
    if os.path.exists(f"OUTPUT/checklist/checklist_{i}_{again}.json"):
        with open(f"OUTPUT/checklist/checklist_{i}_{again}.json", 'r') as f:
            pruned_checklists = json.load(f)
    else:
        pruned_checklists = pruning_checks(aspects, args, f"OUTPUT/checklist/checklist_{i}_{again}.json", checklists)

    if is_in(pruned_checklists, list_prev):
        # os.remove(f"OUTPUT/checklist/checklist_{i}_{again+1}.json")
        stopping_condition=True 
        print('Pruning did not update anything')

    return pruned_checklists, stopping_condition, again

def generate_response(data, args, checklists, i, data_type, again=0):
    '''
    input: data, args, checklists, i-iteration, data_type-train/test
    description:
        - generate response on given dataset
        - post process comments: remove duplicates, and N/A values
        - save the comments to file
    output: cleaned_comments, data
    '''
    # generate response on given dataset
    file = f"OUTPUT/response/{data_type}_{i}_{again}.jsonl"
    if not os.path.exists(file):
        comments, response = gen(data, args, file, checklists)
    else:
        comments = load_saved_comments(i, data_type)
        with open(file, 'r', encoding='utf-8') as f:
            response = [json.loads(line) for line in f]
    print(f"GENERATED Responses on {data_type} data")

    # post process comments: remove duplicates, and N/A values
    cleaned_comments={}
    for d in response:
        for a, comm in d['comments'].items():
            if a not in cleaned_comments:
                cleaned_comments[a] = []
            if d['useful_comments'][a]%3: # [1,2]
                cleaned_comments[a].append(comm)

    for k, v in cleaned_comments.items():
        cleaned_comments[k] = list(set(v))

    # save the comments to file
    if args.save_all:
        print("Writing cleaned comments to file")
        with open(f"OUTPUT/metadata/comments/{data_type}_{i}_{again}.json", 'w') as f:
            json.dump(cleaned_comments, f)
    return cleaned_comments, comments, response

def load_saved_comments(i, data_type):
    '''
    input: i-iteration
    description:
        - load the saved comments from the output file
    output: comments
    '''
    comments={}
    again = 3
    while not os.path.exists(f"OUTPUT/response/{data_type}_{i}_{again}.jsonl"):
        again-=1
    # load the saved comments from the train data output file
    with open(f"OUTPUT/response/{data_type}_{i}_{again}.jsonl", 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    for d in data:
        for key, value in d['comments'].items():
            if key not in comments:
                comments[key] = []
            if d['useful_comments'][key]==3:
                comments[key].append("na") # if not useful then dont add in comments section
            else:
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
            if set(checklists[aspect])!=set(checks):
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
        if 'GT_annotation' in data[0].keys() and args.aspects == None:
            args.aspects = list(data[0]['GT_annotation'].keys())
        if args.aspects==[]: args.aspects = ['na']

        assert args.train_test_size[0]+args.train_test_size[1]<=len(data), "Sum of train and test dataset size should not be more than length of dataset: "+str(len(data))
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
        if 'output' not in data[0].keys():
            for d in data:
                d['output'] = ""
        if 'GT_annotation' not in data[0].keys():
            for d in data:
                d['GT_annotation'] = {}
                for asp in args.aspects:
                    d['GT_annotation'][asp] = -1

        if args.train_test_size[1]==0:
            if args.save_all:
                if data and not os.path.exists(args.train_dataset_path+'_train.jsonl'):
                    with open(args.train_dataset_path+'_train.jsonl', 'w', encoding='utf-8') as f:
                        for d in data:
                            f.write(json.dumps(d, ensure_ascii=False)+'\n')
            return data, []
        if args.train_test_size[0]==0:
            if args.save_all:
                if data and not os.path.exists(args.train_dataset_path+'_test.jsonl'):
                    with open(args.train_dataset_path+'_test.jsonl', 'w', encoding='utf-8') as f:
                        for d in data:
                            f.write(json.dumps(d, ensure_ascii=False)+'\n')
            return [], data
        

        # randomly sample train and test data and save it to a file
        random.seed(42)  # for reproducibility
        
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
    

    args = get_args()
    args.eval = False
    args.resp_comments = 1
    args.stoping_precentage = 100
    if args.aspects!=None:
        args.aspects = ast.literal_eval(args.aspects)
    args.dataset_path = os.path.join("data", args.dataset_path)
    with open(args.dataset_path, 'r', encoding='utf-8') as f: 
        data = [json.loads(line) for line in f]
    
    args.train_test_size = [len(data), 0]
    asserting_inputs(args)
    
    args.train_dataset_path = args.dataset_path.replace(".jsonl", f"_{str(args.train_test_size)}")
    args.train_dataset_size, args.test_dataset_size = args.train_test_size
    
    # initialize output folders and train, test datasets
    train_data, test_data = obtain_test_train_files(args, data)
    
    if not os.path.exists("OUTPUT"):
        os.makedirs("OUTPUT")
    
    if args.save_all:
        folder = ['checklist', 'response', 'metadata']
        for f in folder:
            if not os.path.exists(f"OUTPUT/{f}"):
                os.makedirs(f"OUTPUT/{f}")
        folder = ['comments', 'prune']
        for f in folder:
            if not os.path.exists(f"OUTPUT/metadata/{f}"):
                os.makedirs(f"OUTPUT/metadata/{f}")
        # save args to txt file in metadata folder
        with open("OUTPUT/metadata/args.txt", 'w') as f:
            f.write(str(args))

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
    with open("helper/utils.json", 'w') as f:
        json.dump(description, f)

    args.description = description  # save the description in args
    return args, train_data, test_data


def main():
    args, train_data, test_data = wrapper_args()

    print("STARTING... TASK:", args.task)  
    again=0 # updated checklist
    i=0 # iteration
    cur_aspects = args.aspects.copy()
    
    if args.custom_checklist:
        print("Starting from custom seed checklist")
        with open(args.custom_checklist, 'r', encoding="utf-8") as f:
            seed_checklists = json.load(f)
    else:
        seed_checklists = seed_checks(args, f"OUTPUT/checklist/checklist_{i}_{again}.json")
    list_prev = []
    list_prev.append(seed_checklists) # maintaining list of previous checklists
    checklists = seed_checklists.copy()

    stopping_condition = False
    mapped_aspects = {}
    for i, aspect in enumerate(args.aspects):
        mapped_aspects[i] = aspect

    while not stopping_condition:
        print(f"\n\nITERATION: {i}")
        
        # generate and evaluate responses on train and test data
        responses, cur_aspects, cleaned_comments, stopping_condition = generate_evaluate_response(cur_aspects, mapped_aspects, train_data, test_data, args, checklists.copy(), i, again)
        
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
        again=0
        checklists = updating_checks(cur_aspects, args, cleaned_comments, f"OUTPUT/checklist/checklist_{i}_{again}.json", checklists.copy())
        
        if is_in(checklists, list_prev): 
            stopping_condition=True
            print("Stoping: Checking not updated---")
        else: 
            list_prev.append(checklists.copy())
        
        if args.calling_prune==3: # prune iteratively
            print("PRUNING ITERATIVELY... at", i, 1)
            checklists, stopping_condition, again = pruning_checklist(i, again, cur_aspects, args, checklists, list_prev)
            if again>1: again=1
        if stopping_condition:
            break

    # after reaching stopping condition, prune final checklist and run responses again 
    print()
    if args.calling_prune in [1, 2]: # prune only once at the end
        print("PRUNING ONCE... at", i, again)
        checklists, stopping_condition, again  = pruning_checklist(i, again, cur_aspects, args, checklists, list_prev)
        if not stopping_condition:
            # generate and evaluate responses on train and test data
            responses, cur_aspects, cleaned_comments, stopping_condition = generate_evaluate_response(cur_aspects, mapped_aspects, train_data, test_data, args, checklists, i, again)

            if args.calling_prune==2:
                print()
                print("PRUNING TWICE... at", i, again)
                checklists, stopping_condition, again = pruning_checklist(i, again, cur_aspects, args, checklists, list_prev)
            if not stopping_condition:
                # generate and evaluate responses on train and test data
                responses, cur_aspects, cleaned_comments, stopping_condition = generate_evaluate_response(cur_aspects, mapped_aspects, train_data, test_data, args, checklists, i, again)


    # # after reaching stopping condition, save the final checklist
    print()
    with open(f"output/checklist.json", 'w', encoding="utf-8") as f:
        json.dump(checklists, f, indent=4, ensure_ascii=False)
    # for data_type, resp in responses.items():
    #     with open(f"output/response_{data_type}.jsonl", 'w', encoding="utf-8") as f:
    #         for r in resp:
    #             f.write(json.dumps(r, ensure_ascii=False)+'\n')
    print("Final checklist saved in output folder")

    print("STOPPING...")
    return


if __name__ == "__main__":
    # try:
        main()
    # except Exception as err:
    #     import pdb
    #     pdb.post_mortem()