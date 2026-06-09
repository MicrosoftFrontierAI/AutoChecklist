'''
input file contains jsonl file with following format:
current checklist, comments, task, aspects

Updates the checklist and stores in json file
'''
import json, os, re

from script.prompting import get_response
from script.prompt_utils import load_prompt, format_checklist


def parse_response(response, checklist_aspect, args):
    '''parses response and updates checklist'''
    resps = response.replace("\\n", "").replace('\n', '').split("---")[1:]
    flag=0
    op=[]
    result=[]
    cur=len(checklist_aspect)+1
    for resp in resps:
        splitted = resp.split(': ')
        ques = splitted[-1].strip()
        if ques!="":
            if (resp.count(': ')==2 and int(splitted[1].strip())==cur):
                # ---question id: 1: question...
                flag=1
            if resp.count(': ')==1: 
                try:
                    # ---1: question...
                    if int(splitted[0].strip())==cur:
                        flag=1
                except Exception as e:
                    if ques not in checklist_aspect:
                        flag=1

            cur+=1
            op.append(ques)
    
    result = op
    if args.update_version==2 or flag: # appended
        # print("Appending ...")
        result.extend(checklist_aspect)
    result = list(set(result)) # remove duplicates
    return result


def updating_checks(aspects, args, comments, output_filepath, cur_checklist, prompt_filepath="update.txt"):
    '''
    input: aspects, args, comments, output_filepath, checklist, prompt_filepath
    description:
        - Loads any saved checklist, if exists
        - Updates the checklist
        - Saves the updated checklist to file, if args.save_all
    output: updated checklist
    '''
    if os.path.exists(output_filepath): # Load from saved file if exists
        return json.load(open(output_filepath, 'r', encoding="utf-8"))
    
    # Load prompt
    with open(os.path.join("helper", "prompts", prompt_filepath), 'r', encoding="utf-8") as f:
        PROMPT = f.read()
    PROMPT = load_prompt(PROMPT, args)[0]
    
    prompt_updated = PROMPT
    # history.append(prompt_updated.split("<user>")[0].strip())
    aspect_list = list(cur_checklist.keys())
    check_list = list(cur_checklist.values())
    checklist = {aspect: [] for aspect in aspect_list}
    for aspect, checks in zip(aspect_list, check_list):
        if aspect in aspects:
            prompt_updated = prompt_updated.replace("{aspect}", aspect).replace("{description}", args.description[args.task]['aspects'][aspect.lower()]).replace("{checklist}", format_checklist(checks)).replace("{comments}", str(comments[aspect.lower()]))
            
            # check for any missed parameters to replace in the prompt
            pattern = r'\{[^}]*\}'
            matches = re.findall(pattern, prompt_updated)
            assert len(matches)==0, f"Unreplaced parameters in the prompt: {matches}"
        
            resp, _, _ = get_response([], prompt_updated, args.llm, stop="<assistant>", max_tokens=3500)   
            resp = parse_response(resp[0], checks, args)
        else:
            resp = checks
        checklist[aspect.lower()] = resp # load previous

    if args.save_all:
        # print("Writing updated checklist ...")
        with open(output_filepath, 'w', encoding="utf-8") as f:
            json.dump(checklist, f, ensure_ascii=False, indent=4)
    return checklist