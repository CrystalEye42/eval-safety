import json

def obtain_stats(f1, f2):
    results = [0, 0, 0, 0]
    for k in f1.keys():
        if type(f1[k]) != bool:
            sub_results = obtain_stats(f1[k], f2[k])
            results = [a + b for a, b in zip(results, sub_results)]
        else:
            i = 2 * int(f1[k]) + int(f2[k])
            results[i] += 1

            if f1[k] != f2[k]:
                print(k)
                print(f1[k])
            
    return results

def get_json(file_name):
    with open(file_name, 'r') as f:
        file = json.load(f)
    return file


def print_results(results):
    print("Both T: ", results[3])
    print("First T:", results[2])
    print("First F:", results[1])
    print("Both F: ", results[0])

unpruned = get_json('evaluations/Llama-2-7b-chat-hf/jailbreak.json')
pruned = get_json('evaluations/unstructured_50/jailbreak.json')

print_results(obtain_stats(unpruned, pruned))
