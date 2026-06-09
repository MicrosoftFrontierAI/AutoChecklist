'''
generates response on dataset using checklist
'''
import json, tqdm, os, re, numpy as np

from script.prompting import get_response
from script.prompt_utils import load_prompt, format_checklist


def transform(list_json):
    '''converts list of json to string, making Explanations more readable'''
    if type(list_json)!=dict or "<p>" not in list_json[0].values(): return list_json
    t="\n"
    for i in list_json:
        for k, v in i.items():
            v=v.replace('<p>', "").replace('</p>\n', "").replace('<ul>', "").replace('</ul>', "").replace('<li>', "- ").replace('</li>', "")
            t+=f"# {k.upper()}: \n{v}\n"
    return t

def agg_score(args, responses, error):
    '''aggregates responses to a single score'''
    if error: return 0
    if args.resp_type:
        '''responses: list of 1-5, 1-5 rating...'''
        resps=[]
        for resp in responses:
            try: resp = int(resp)
            except: # extract the interger part
                resp = [int(s) for s in resp.split() if s.isdigit()]
            resps.append(resp)
        op = sum(resps)/len(resps) # average rating from 1-5
    else:
        '''responses: list of "yes", "no"'''
        yes = responses.count("yes")
        no = responses.count("no")
        score = yes/(yes+no) # score from 0-1
        op = 1 + score*4 # normalize from 1-5
    # scale from 1-5 to 0-3
    if args.task=="instruct_excel":
        op = np.round(3*(op-1)/4, 3)
    return op

def parse_response(args, response, comments):
    '''parses response and aggregates score'''
    response = response.lower().replace("\\n", "\n")
    resp=[]
    error=False
    try:
        temp = response.split("comments:")[0].split("---")[1:]
        for i in temp:
            if ": " not in i: continue
            resp.append(i.split(": ")[-1].strip())
        comments.append(response.split("comments:")[-1].strip())
    except: # if no comments
        try:
            temp = response.split("---")[1:]
            for i in temp:
                if ": " not in i: continue
                resp.append(i.split(": ")[-1].strip())
            comments.append("NA")
        except Exception as e:
            print(e)
            error=True
    if resp == []: error=True

    return resp, agg_score(args, resp, error), comments, error

def averaging_results(output, num_checks): 
    '''
    averages results if multiple responses are given
    output: list of responses, aggregated score, comments, error
    num_checks: number of checks in checklist
    '''
    resp, _, comments, error = output
    num_sugg = len(resp)//num_checks
    new_resp = []

    for i in range(num_checks):
        temp = 0
        for j in range(num_sugg):
            temp += (1 if resp[i+j*num_checks].lower() == "yes" else 0)
        new_resp.append(str(temp/num_sugg))
    return new_resp, _, comments, error

def gen(datas, args, output_filepath, checklist, prompt_filepath="gen.txt", filter=True):
    '''
    input: args, output_filepath, checklist, prompt_filepath
    description: 
        - loads prompt and dataset
        - generates responses using checklist
        - aggregates responses and saves to output_filepath
    output: comments, response dataset
    '''
    # Load prompt
    with open(os.path.join("helper", "prompts", prompt_filepath), 'r', encoding="utf-8") as f:
        PROMPT = f.read()
    PROMPTS = load_prompt(PROMPT, args)
    additional_col = ["weighted_score", "aggregated_score", "full_response", "comments", "useful_comments"]
    
    for wo_comment,PROMPT in enumerate(PROMPTS):
        if wo_comment: # args.resp_comments==2
            # print("Generating responses without comments ...")
            wo = "_wocomments"
            additional_col = [a+wo for a in additional_col[:-1]]
            total_comments = comments.copy()
        # else:
            # print("Generating responses")

        comments = {}
        for aspect, checks in checklist.items():
            PROMPT_ASPECT = PROMPT.replace("{aspect}", aspect).replace("{description}", args.description[args.task]['aspects'][aspect.lower()]).replace("{checklist}", format_checklist(checks))
            comments[aspect.lower()] = []
            # history.append(prompt_updated.split("<user>")[0].strip()) # concatenate responses in chat mode
            
            for i, data in tqdm.tqdm(enumerate(datas)):
                # initialize other response columns
                for col in additional_col: 
                    if col not in data.keys(): data[col] = {}
                
                max_tokens=250
                if args.resp_type: max_tokens+=100
                added=500
                transformed = transform(data["input"])
                op = ""
                
                while True:
                    if 'output' in data.keys() and data['output']!="":
                        op = 'OUTPUT: '+data["output"]
                    prompt_updated = PROMPT_ASPECT.replace("{input}", transformed).replace("{output}", op)

                    # check for any missed parameters to replace in the prompt
                    pattern = r'\{[^}]*\}'
                    matches = re.findall(pattern, prompt_updated)
                    # assert len(matches)==0, f"Unreplaced parameters in the prompt: {matches}"

                    resp, _, _ = get_response([], prompt_updated, args.llm, stop="<assistant>", max_tokens=max_tokens+added)

                    # currently only considering 1st completion and not multi response
                    output = parse_response(args, resp[0], comments[aspect.lower()])
                    if not output[-1] and len(checks)==len(output[0]): break
                    if len(checks)!=len(output[0]): 
                        if len(output[0])%len(checks)==0: 
                            output = averaging_results(output, len(checks))
                            if len(output[0])==len(checks):  break
                        print(len(checks), len(output[0]))
                        print("redoing-response number is not correct size as checklist")
                        added+=1
                        if added-500>5:
                            print(f"FOR {i}, tried 5 times still error")
                            br
                    else:
                        added+=100
                        print("redoing with increased tokens-whole completion not generated")
                resp, compiled, comments[aspect.lower()], _ = output
                data[additional_col[0]][aspect.lower()] = 0
                data[additional_col[1]][aspect.lower()] = compiled
                data[additional_col[2]][aspect.lower()] = resp
                if not wo_comment and args.resp_comments: 
                    data["comments"][aspect.lower()] = comments[aspect.lower()][i]

    if filter: # questioning LLM if comment addresses issue already covered in checklist
        datas = filter_comments(datas, checklist, args)

    if args.save_all:
        # print("writing responses ...")
        with open(output_filepath, 'w', encoding="utf-8") as f:
            for data in datas:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    if len(PROMPTS)==1: return comments, datas
    return total_comments, datas


def filter_comments(datas, checklist, args):
    '''filtering comments if they are already covered in checklist'''
    # print("Filtering non useful comments ...")
    # load filtering prompt
    with open(os.path.join("helper", "prompts", "filter.txt"), 'r', encoding="utf-8") as f:
        PROMPT = f.read()

    for aspect, checks in checklist.items():
        prompt = PROMPT.replace('{checklist}', format_checklist(checks))
        history = []
        tokens=250
        for data in datas:
            if aspect not in data['comments'] or data["comments"][aspect]=="na": 
                data["useful_comments"][aspect] = 0
                continue
            prompt_used = prompt.replace("{comment}", data["comments"][aspect]).replace('{answer}', str(data['full_response'][aspect]))
            tries=0
            while(tries<8):
                tries+=1
                resp, _, history = get_response(history, prompt_used, args.llm, stop="<assistant>", max_tokens=tokens+tries*50)
                if "RESPONSE:" in resp[0]: break
            data["useful_comments"][aspect] = parsing_resp(resp[0])
    
    return datas

def parsing_resp(resp):
    '''parsing response to get useful comments'''
    st = ['\n', ']', ']', '`']
    OP = resp.split('RESPONSE:')[1]
    
    for s in st:
        OP = OP.replace(s, '')
    if OP.strip().isdigit(): return int(OP.strip())
    if OP[1:].strip().isdigit(): return int(OP[1:].strip())
    return int(OP.strip())