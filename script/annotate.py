'''
FOR INFERENCING IN EVALUATION TOOLS OVER ANY NEW BENCHMARK:

- uses the checklist to generate annoations on dataset
'''
from argparse import ArgumentParser
import json, os

from script.evaluation.gen import gen

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default="translation", help="Task to perform evaluation on, [summarization, translation, instruct_excel, formual_explaination]")
    parser.add_argument("--dataset_path", type=str, default="example_translation.jsonl", help="Path to file containing the dataset in data folder")
    
    parser.add_argument("--output_path", type=str, default="output/annotated.jsonl", help="Path to save the annotated dataset")
    parser.add_argument("--custom_checklist", type=str, default="output/checklist.json", help="Path to manually curated seed checklist to set as starting point")
    parser.add_argument("--llm", type=str, help="Which LLM (chat/completion) endpoint from your LLM provider is to be used for pipeline. A model name containing 'chat' routes to the chat endpoint; examples: gpt-4o-chat-completions, gpt-4-1106")
    parser.add_argument("--dataset_size", type=int, default=None, help="Total number of initial-input data to annotate, None: means for all dataset")

    return parser.parse_args()

def wrapper_args():
    '''
    description:
        - wrapper function to assert and initialize the parameters and folders
        - read the dataset
    output: args, description, input_data
    '''
    # defining helper functions
    def asserting_inputs(args):
        '''
        input: args
        description:
            - assert the parameters are of correct type
        '''
        # asserting parameters are of correct type
        assert os.path.exists(args.dataset_path), "Dataset file not found in data folder"
        assert os.path.exists(args.custom_checklist), "Seed checklist file not found"
        assert args.output_path.endswith(".jsonl"), "Output path should end with .jsonl, annotation stored in jsonl format"

        with open(args.custom_checklist, 'r', encoding="utf-8") as f:
            checklist = json.load(f)
        args.aspects = list(checklist.keys())
        for aspect in checklist:
            assert len(aspect)>0, "Seed checklist can not be empty"
            for seed in checklist[aspect]:
                assert len(seed)>0, "Seed checklist can not be empty"
        for aspect in args.aspects:
            assert aspect.lower() in checklist, f"Aspect {aspect} not found in seed checklist"

        with open(args.dataset_path, 'r', encoding='utf-8') as f: 
            data = [json.loads(line) for line in f]
        assert args.dataset_size<=len(data), "Input dataset size can not be more than length of dataset: "+str(len(data))
        assert args.dataset_size>0, "Input dataset size should be more than 0"
        return 
    
    def obt_dataset(args, data):
        '''
        inupt: args, dataset
        description: 
            - obtain the test and train files from the dataset
            - randomly sample the dataset and save it to a file(if args.save_all)
        output: input_data
        '''
        input_data = data[:min(args.dataset_size, len(data))]
        if input_data and not os.path.exists(args.saved_dataset_path):
            with open(args.saved_dataset_path, 'w', encoding='utf-8') as f:
                for d in input_data:
                    f.write(json.dumps(d, ensure_ascii=False)+'\n')
        
        return input_data
    

    args = get_args()
    args.eval = False
    args.save_all = True
    args.resp_comments = 0
    args.resp_type = 0
    args.metachecklist = "helper/metachecklist.txt"
    args.prune_version = 3
    args.update_version = 1
    args.seed_w_sample = 0
    # if args.aspects!=None:
    #     args.aspects = ast.literal_eval(args.aspects)
    args.dataset_path = os.path.join("data", args.dataset_path)
    with open(args.dataset_path, 'r', encoding='utf-8') as f: 
        data = [json.loads(line) for line in f]

    if args.dataset_size == None:
        args.dataset_size = len(data)
    args.saved_dataset_path = args.dataset_path.replace(".jsonl", f"_{str(args.dataset_size)}.jsonl")
    asserting_inputs(args)
    
    
    # initialize output folders and train, test datasets
    input_data = obt_dataset(args, data)
    
    if not os.path.exists("OUTPUT"):
        os.makedirs("OUTPUT")

    # load previously saved task and aspect descriptions
    with open("helper/utils.json", 'r') as f:
        description = json.load(f)
    args.description = description  # save the description in args
    return args, input_data


def clean_gen(input_data, args, output_path, checklists):
    gen(input_data, args, output_path, checklists)
    with open(output_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    gen_data = []

    for d in data:
        d.pop('weighted_score')
        d.pop('comments')
        d.pop('useful_comments')
        gen_data.append(d)
    with open(output_path, 'w', encoding='utf-8') as f:
        for d in gen_data:
            f.write(json.dumps(d, ensure_ascii=False)+'\n')
    return


def main():
    args, input_data = wrapper_args()

    print("For the given task of:", args.task)  
    print("Loading checklist")
    with open(args.custom_checklist, 'r', encoding="utf-8") as f:
        checklists = json.load(f)

    if os.path.exists(args.saved_dataset_path):
        print("Removing already present annotations")
        os.remove(args.saved_dataset_path)
        
    print("Starting to annotate")
    clean_gen(input_data, args, args.output_path, checklists)
    print("Done...")
    return


if __name__ == "__main__":
    # try:
        main()
    # except Exception as err:
    #     import pdb
    #     pdb.post_mortem()