import json, random
map_resp_type = {0: "boolean", 1: "rating-mentioned", 2: "rating"}
def format_checklist(checks):
    '''Formats the checklist for display'''
    return "\n".join([f"{i+1}: {check}" for i,check in enumerate(checks)])

# These are components in the prompt: for variations with and without these parameters
comments_1 = '''- If you spot anything else that is interesting property for evaluation, and that is NOT included in the current checklist, then mention that in the comments. 
    - An interesting property could be a wrong or correct pattern, a discrepancy, or anything that stands out to you.
    - Comments cannot be related to the answers to the questions.'''
comments_2 = '''- Add the comments add in the format: "comments: comment"
        - If in case there is no comments to add, then respond with: "comments: NA"
    - Ensure that comments do not repeat properties already covered by the checklist.'''
key_comp = "KEY COMPONENT: {aspect} - {description}"
metachecklist_holder ="- Additionally, keep the follwing tips in mind when you are generating the checklist:\n{metachecklist}"

mentioned = '''Where, 1 means "Strongly disagree",
    2 means "Disagree",
    3 means "Neither agree or disagree",
    4 means "Agree",
    5 means "Strongly agree".'''
resp_type_1 = {"boolean": '''is a Boolean "Yes" or "No" question.''',
"rating": '''gives a rating inputs on Likert scale of 1-5.'''}
resp_type_2 = {"boolean": '''- Each question must be answerable with "Yes" or "No".
- Phrase question so that a "Yes" answer indicates a positive outcome.''',
"rating": '''- Each question must be answerable with rating from 1-5 scale.
    {mentioned}
- Phrase question so that a 5 rating indicates a positive outcome.'''}
resp_type_3 = {"boolean": '''is a Boolean response 
question that aligns with the provided rules.''',
"rating": '''gives a rating response on the 
Likert scale of 1-5, that aligns with the provided rules.'''}
resp_type_4 = {"boolean": '''- Respond to each question with "Yes" or "No", while aiming for minimal subjectivity.''',
"rating": '''- Respond to each question with scores from 1-5 rating, while aiming for minimal subjectivity.
    {mentioned}'''}

prune = {
    1: '''- STEP 1: Sequentially arrange the current checklist, which each question in order of similarity with one another.
- STEP 2: Remove redundant checks that coincide with same criteria of evaluation, while retaining one of them. 
This retained question should have broader coverage of crietria compared to the ones that are removed.
- STEP 3: Cluster the similar checks together under a common summarizing criteria.
- STEP 4: For clusters with multiple similar checks, club and combine them to get a reduced amount of summarizing checks. 
This summizing checks should cover all aspects of criterias that each individual question was initially capturing.''',
    2: '''STEP 1: Cluster the similar checks together under a common summarizing criteria.
STEP 2: For clusters with multiple similar checks, club and combine them to get a reduced amount of summarizing checks. 
This summizing checks should cover all aspects of criteria that each individual question was initially capturing.''',
    3: '''- You need to remove all redundant/duplicate checks within the checklist. 
Ensure each question is unique and checking for totally different criteria.
- Then you need to summarize for any highly related checks that question for the same criteria.
Ensure each question covers to evaluate over one criteria only.'''
}
update = {
    1: '''- Keep the current checklist in mind as the starting point.
- You need to come up with additional criteria/checks to evaluate for the task, based the additional comments.
- Omit generating any criteria that is already existing/present within the current checklist.
- Just output the new modified criteria that needs to be appended to the current checklist, and not the current checklist.''',
    2: '''- Keep the current checklist in mind as the starting point.
- You need to come up with additional questions to evaluate the task, based the comments.
- Omit generating any criteria that is already existing/present within the current checklist.'''
# next appending checklist only returned
}


samples_1 = '''- Use the input samples to direct the generation of checklist by observing what could 
possibly go wrong in such inputs given the task for evaluation of quality.'''

def load_prompt(PROMPT, args):
    with open(args.dataset_path, 'r', encoding='utf-8') as f: 
        data = [json.loads(line) for line in f]

    resp_type = map_resp_type[args.resp_type]
    PROMPT = PROMPT.replace("{task}", args.description[args.task]['description'])
    if args.aspects==['na']:
        PROMPT = PROMPT.replace("{key component}", "")
    else:
        PROMPT = PROMPT.replace("{key component}", key_comp)
    PROMPT = PROMPT.replace("{resp_type_1}", resp_type_1[resp_type.split('-')[0]])
    PROMPT = PROMPT.replace("{resp_type_2}", resp_type_2[resp_type.split('-')[0]])
    PROMPT = PROMPT.replace("{resp_type_3}", resp_type_3[resp_type.split('-')[0]])
    PROMPT = PROMPT.replace("{resp_type_4}", resp_type_4[resp_type.split('-')[0]])
    if 'mentioned' in resp_type:
        PROMPT = PROMPT.replace("{mentioned}", mentioned)
    else:
        PROMPT = PROMPT.replace("{mentioned}", "")


    def extract_samples(data, sample_type, aspect, num=3):
        '''
        input: data, sample_type
        output: samples
        '''
        random.seed(42)
        data = data.sort(key=lambda x: x["GT_annotation"][aspect.lower()])
        samples = []
        if sample_type == 1: # diverse
            samples.append(data[0])
            samples.append(data[-1])
            samples.append(data[len(data)//2])
        elif sample_type == 2: # random
            samples = random.sample(data, num)
        elif sample_type == 3: # high
            samples = data[-num:]
        else: # low
            samples = data[:num]

        resp = []
        for i,sample in enumerate(samples):
            if sample['output'] == "":
                resp.append(f"SAMPLE-{i+1}:\n INPUT: {sample['input']}")
            else:
                resp.append(f"SAMPLE-{i+1}:\n INPUT: {sample['input']}\n OUTPUT: {sample['output']}")
        return "\n".join(resp)

    if args.metachecklist:
        with open(args.metachecklist, 'r', encoding="utf-8") as f:
            metachecklist = f.read().split("\n")
        PROMPT = PROMPT.replace("{metachecklist_holder}", metachecklist_holder)
        PROMPT = PROMPT.replace("{metachecklist}", format_checklist(metachecklist))

    PROMPT = PROMPT.replace("{prune}", prune[args.prune_version])
    PROMPT = PROMPT.replace("{update}", update[args.update_version])
    
    if args.seed_w_sample:
        PROMPT = PROMPT.replace("{samples_1}", samples_1)
        PROMPT = PROMPT.replace("{samples}", extract_samples(data, args.seed_w_sample, args.aspects[0]))
    else:
        PROMPT = PROMPT.replace("{samples_1}", "")
        PROMPT = PROMPT.replace("{samples}", "")

    
    if not args.resp_comments:
        PROMPT = PROMPT.replace("{comments_1}", "")
        PROMPT = PROMPT.replace("{comments_2}", "")
        return [PROMPT]
    elif args.resp_comments==1:
        PROMPT = PROMPT.replace("{comments_1}", comments_1)
        PROMPT = PROMPT.replace("{comments_2}", comments_2)
        return [PROMPT]
    else:
        PROMPT_w_comments = PROMPT.replace("{comments_1}", comments_1)
        PROMPT_w_comments = PROMPT_w_comments.replace("{comments_2}", comments_2)

        PROMPT_wo_comments = PROMPT.replace("{comments_1}", "")
        PROMPT_wo_comments = PROMPT_wo_comments.replace("{comments_2}", "")
        return [PROMPT_w_comments, PROMPT_wo_comments]