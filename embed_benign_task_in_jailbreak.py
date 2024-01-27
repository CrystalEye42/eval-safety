import json
from ruamel.yaml import YAML
yaml = YAML(typ='safe')
yaml.preserve_quotes = True
yaml.default_flow_style = False
yaml.allow_unicode = True


def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.load(file)
    return data


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


# Reading the YAML files from disk
tasks = read_json_file('./benign_tasks.json')
prompts = read_yaml_file('./malicious_prompts.yaml')

output = {}
for tag, prompt in prompts.items():
    output[tag] = []
    for task in tasks.values():
        new_prompt = prompt[0].replace("[INSERT TASK HERE]", "'" + task + "'")
        output[tag].append(new_prompt)
output_file_path = 'benign_integrated.yaml'
with open(output_file_path, 'w', encoding='utf-8') as file:
    yaml.dump(output, file)
