'''
input file contains jsonl file with following format:
seed checklist, task, key components

Prunes the checklist by removal and summarization of checks and stores in json file
'''
import json, os, re

from script.prompting import get_response
from script.prompt_utils import load_prompt, format_checklist


def parse_response(response, checklists, aspects, f=None):
    '''Parses the response and updates the checklist'''
    response = response.replace("\\n", "\n")
    permutations = ["CHECKLIST", "Checklist", "checklist"]
    for p in permutations:
        if p in response:
            resps = response.split(p)[1].split("\n")
            break

    try:
        # resps = response.replace("\\n", "\n").split("UPDATED CHECKLIST")[1].split("\n")
        checklists[aspects] = [resp.split(': ')[-1].strip() for resp in resps if ": " in resp and resp.split(': ')[-1].strip()!=""]
    except:
        try:
            # resps = response.replace("\\n", "\n").split("Optimized Checklist")[1].split("\n")
            temp = [resp.split(': ')[-1].strip() for resp in resps if ": " in resp and resp.split(': ')[-1].strip()!=""]
            # COMMENT IF USING NOT-MENTIONED RATING RESPONSE(2)
            if " - " in temp[0]:
                temp = [resp.split(' - ')[-1].strip() for resp in temp]
            ##################
            if temp==[]:
                temp = [resp.split('. ')[-1].strip() for resp in resps if ". " in resp and resp.split('. ')[-1].strip()!=""]
            checklists[aspects] = temp
        except:
            try:
                resps = response.split(f"{f}:")[1].split("\n")
                checklists[aspects] = [resp.split(': ')[-1].strip() for resp in resps if ": " in resp and resp.split(': ')[-1].strip()!=""]
                if checklists[aspects] == []:
                    checklists[aspects] = [resp.split(': ')[-1].strip() for resp in resps if ": " in resp and resp.split(': ')[-1].strip()!=""]
            except Exception as e:
                print(e)
                print(response)
                br
    checklists[aspects] = list(set(checklists[aspects]))
    return checklists


def pruning_checks(aspects, args, output_filepath, checklist, prompt_filepath="prune.txt"):
    '''
    input: aspects, args, output_filepath, checklist, prompt_filepath
    description:
        - Loads any saved checklist, if exists
        - Updates the checklist
        - Saves the updated checklist to file, if args.save_all
    output: updated checklist
    '''
    f="STEP 2" if args.update_version==2 else "STEP 4"
    if args.update_version==3:
        f="STEP"
    if os.path.exists(output_filepath): # Load from saved file if exists
        return json.load(open(output_filepath, 'r', encoding="utf-8"))

    # Load prompt
    with open(os.path.join("helper", "prompts", prompt_filepath), 'r', encoding="utf-8") as f:
        PROMPT = f.read()
    PROMPT = load_prompt(PROMPT, args)[0]

    prompt_updated = PROMPT
    metadata = {}
    # history.append(prompt_updated.split("<user>")[0].strip())
    for aspect, checks in checklist.items():
        if aspect in aspects:
            try_=3
            added=0
            while try_>0:
                prompt_updated = prompt_updated.replace("{aspect}", aspect).replace("{description}", args.description[args.task]['aspects'][aspect.lower()])
                prompt_updated = prompt_updated.replace("{checklist}", format_checklist(checks))
                
                # check for any missed parameters to replace in the prompt
                pattern = r'\{[^}]*\}'
                matches = re.findall(pattern, prompt_updated)
                assert len(matches)==0, f"Unreplaced parameters in the prompt: {matches}"

                resp, _, _ = get_response([], prompt_updated, args.llm, stop="<assistant>", max_tokens=3500+added)   
                metadata[aspect.lower()]=resp[0]    
                if "checklist:" in resp[0].lower() or f"{f}:" in resp[0] or "optimized checklist" in resp[0].lower():
                    output = parse_response(resp[0], checklist, aspect.lower(), f)
                    # print("Pruning done for", aspect)
                    break
                try_-=1
                print("Retrying...", try_)
                added+=100
            checklist = output
        else:
            checklist[aspect.lower()] = list(set(checks))
    
    if args.save_all:
        # print("Writing updated checklist ...")
        with open(output_filepath, 'w', encoding="utf-8") as f:
            json.dump(checklist, f, ensure_ascii=False, indent=4)
        with open(output_filepath.replace('checklist', 'metadata').replace('metadata/', 'metadata/prune/'), 'w', encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
    return checklist