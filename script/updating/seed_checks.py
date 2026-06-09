'''
Generates seed checklist
'''
import json, os, re

from script.prompting import get_response
from script.prompt_utils import load_prompt


def parse_response(response, checklists, aspects):
    '''
    input: response, checklists, aspects
    output: updated checklists for the next aspects
    '''
    resps = response.replace("\\n", "\n").split("---")[1:]
    checklists[aspects] = [resp.split(': ')[1].strip() for resp in resps]
    checklists[aspects] = list(set(checklists[aspects]))
    return checklists


def seed_checks(args, output_filepath, prompt_filepath="seed.txt"):
    '''
    input: args, output_filepath, prompt_filepath
    description: 
        - Loads any saved checklist, if exists
        - Generates seed checklist
        - Saves the seed checklist to file, if args.save_all
    output: seed checklist
    '''
    # Load from saved file if exists
    if os.path.exists(output_filepath):
        return json.load(open(output_filepath, 'r', encoding="utf-8"))
    
    # Load prompt
    with open(os.path.join("helper", "prompts", prompt_filepath), 'r', encoding="utf-8") as f:
        PROMPT = f.read()
    PROMPT = load_prompt(PROMPT, args)[0]

    # Generate the seed checklist
    checklists = {}
    history = []
    tokens = 250
    for aspect in args.aspects:
        prompt_updated = PROMPT.replace("{aspect}", aspect).replace("{description}", args.description[args.task]['aspects'][aspect.lower()])
        
        # check for any missed parameters to replace in the prompt
        pattern = r'\{[^}]*\}'
        matches = re.findall(pattern, prompt_updated)
        assert len(matches)==0, f"Unreplaced parameters in the prompt: {matches}"
        
        resp, _, history = get_response(history, prompt_updated, args.llm, stop="<assistant>", max_tokens=tokens)
        checklists = parse_response(resp[0], checklists, aspect.lower())
        tokens += 150

    # Save the seed checklist to file
    if args.save_all:
        with open(output_filepath, 'w', encoding="utf-8") as f:
            json.dump(checklists, f, indent=4, ensure_ascii=False)
    
    return checklists