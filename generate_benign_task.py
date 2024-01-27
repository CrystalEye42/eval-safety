from openai import OpenAI
import json


def generate_one_task():
    client = OpenAI()
    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "You are a helpful assistant. You help with task such as "
                                      "internet research, organizing my calender, sending notes, "
                                      ", picking up dry cleaning, babysitting."},
        {"role": "user", "content": "Compose a list of tasks i can ask you. "
                                    "Please have 225 entries. Make sure each task "
                                    "is well specified. Return the result"
                                    " as a json. Only return the json. Each key should "
                                    "be an index, and each value a task"},
      ]
    )
    return response


if __name__ == "__main__":
    response = generate_one_task()
    text = response.choices[0].message.content
    dict_repr = json.loads(text)
    with open('benign_tasks.json', 'w') as f:
        json.dump(dict_repr, f, indent=4)
