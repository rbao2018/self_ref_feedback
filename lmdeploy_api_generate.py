import os
import jsonlines
import os.path as osp
import argparse
import time
import threading
import random
import requests
from typing import List
from queue import Queue
import concurrent.futures
import copy

from lmdeploy.serve.openai.api_client import APIClient

def parse_args():
    parser = argparse.ArgumentParser(description="Lmdeploy for LLM feedback generate.")
    parser.add_argument("--template", type=str, default="Human:{sys_prompt}{instruction}\n")
    parser.add_argument("--user_input", type=str, default="A:{output_a}\nB:{output_b}")
    parser.add_argument("--datasets", nargs='*', default=None, help="Path to the dataset.")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to the dataset directory.")
    parser.add_argument("--task_name", type=str, default='dialog_inference')
    parser.add_argument("--client_port", type=int, default=10080)
    parser.add_argument("--num_clients", type=int, default=8)
    parser.add_argument("--model_class", type=str, default="llama2")
    parser.add_argument("--model_name", type=str, default="llama2")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=2048, help="Output length")
    parser.add_argument("--round_num", type=int, default=2)
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--llm_feedback", action="store_true")
    parser.add_argument("--second_run", action="store_true")
    parser.add_argument("--num_prompts", type=int, default=12800000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--file_index", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--output_file", type=str, default="output.jsonl")
    parser.add_argument("--output_key", type=str, default="ours_7b_a")
    parser.add_argument("--second_pair_key", type=str, default="ours_7b_b")
    args = parser.parse_args()
    return args

class Engine:
    def __init__(self, args, server_addrs: List[str], **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.server_addrs = server_addrs
        self.api_clients = [APIClient(addr) for addr in server_addrs]
        self.generation_config = {
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "max_tokens": self.args.max_tokens
        }
        self.generate_text = Queue()
    
    def clean_generate_text(self):
        self.generate_text = Queue()

    def chose_available_client(self, index):
        for i in range(len(self.api_clients)):
            client = self.api_clients[(i + index) % len(self.api_clients)]
            try:
                _ = client.available_models
                return client
            except Exception as e:
                time.sleep(0.05)
                continue
        return None

    def dialog_inference(self, data, round_num=0):
        data_index = data.get("index", 0)
        chose_client = self.chose_available_client(data_index)
        try:
            if "conversations" not in data.keys():
                instruction_input = data['instruction']
                inputs = [{"role":"user", "content": self.args.template.format(instruction_input)}]
                model_output = ""
                for message in chose_client.chat_completions_v1(
                    model = self.args.model_class,
                    messages=inputs,
                    stream=True,
                    **self.generation_config):
                    if len(message.choices) > 0 and message.choices[0].delta.content is not None:
                        model_output += message.choices[0].delta.content
                data.update({
                    f'{self.args.model_name}_{round_num}': model_output
                })
                self.generate_text.put(data, timeout=8)
            else:
                inputs = []
                for index, conversation in enumerate(data['conversations']):
                    content = self.args.template.format(conversation["instruction"])
                    inputs.append({"role":"user", "content": content})
                    model_output = ""
                    for message in chose_client.chat_completions_v1(
                        model = self.args.model_class,
                        messages=inputs,
                        stream=True,
                        **self.generation_config):
                        if len(message.choices) > 0 and message.choices[0].delta.content is not None:
                            model_output += message.choices[0].delta.content
                    conversation.update({
                        f'{self.args.model_name}_{round_num}': model_output
                    })
                    inputs.append({"role":"assistant", "content": model_output})
                    if index >= 8:
                        break
                self.generate_text.put(data, timeout=8)
        except Exception as e:
            print(f"error:{e}", flush=True)    
        if data_index % 100 == 0:
            print(f"INDEX:{data_index} DATA:{data}", flush=True)

    def direct_feedback(self, data, round_num=0):
        intext = ''; output_a = ''
        data_index = data.get("index", 0)
        chose_client = self.chose_available_client(data_index)
        for try_times in range(8):
            try:
                for index, conversation in enumerate(data['conversations']):
                    sys_prompt = data['sys_prompt'] if index == 0 and 'sys_prompt' in data.keys() else ''
                    new_conversation = copy.deepcopy(conversation)
                    new_conversation['sys_prompt'] = sys_prompt
                    if index == len(data['conversations']) - 1:
                        output_a = new_conversation[self.args.output_key]
                    intext += self.args.template.format_map(new_conversation)
                    if index < len(data['conversations']) - 1:
                        intext += "Assistant:" + new_conversation[self.args.output_key] + "\n"
                if "[INST]" in intext or "[/INST]" in intext:
                    break
                if round_num % 2 == 0:
                    input_dict = {
                        'output_a': output_a,
                        'output_b': data[self.args.second_pair_key]
                    }
                else:
                    input_dict = {
                        'output_a': data[self.args.second_pair_key],
                        'output_b': output_a
                    }
                user_input = self.args.user_input.format_map(input_dict)
                inputs = [{"role":"user", "content": intext + user_input}]
                model_output = ""
                for message in chose_client.chat_completions_v1(
                    model = self.args.model_class,
                    messages=inputs,
                    stream=False,
                    **self.generation_config):
                    model_output = message['choices'][0]['message']['content'] 
                data.update({f'{self.args.model_name}_{round_num}': model_output})
                self.generate_text.put(data, timeout=8)
                break
            except Exception as e:
                print(f"error:{e} with try time {try_times}", flush=True)
                time.sleep(32)
        if data_index % 100 == 0:
            print(f"INDEX:{data_index}\nDATA:{data}", flush=True)

    def reference_feedback(self, data, round_num=0):
        intext = ''; output_a = ''
        data_index = data.get("index", 0)
        chose_client = self.chose_available_client(data_index)
        for try_times in range(8):
            try:
                for index, conversation in enumerate(data['conversations']):
                    sys_prompt = data['sys_prompt'] if index == 0 and 'sys_prompt' in data.keys() else ''
                    new_conversation = copy.deepcopy(conversation)
                    new_conversation['sys_prompt'] = sys_prompt
                    if index == len(data['conversations']) - 1:
                        output_a = new_conversation[self.args.output_key]
                    intext += self.args.template.format_map(new_conversation)
                    if index < len(data['conversations']) - 1:
                        intext += "Assistant:" + new_conversation[self.args.output_key] + "\n"
                if "[INST]" in intext or "[/INST]" in intext:
                    break
                inputs = [{"role":"user", "content": intext}]
                model_output = ""
                for message in chose_client.chat_completions_v1(
                    model = self.args.model_class,
                    messages=inputs,
                    stream=False,
                    **self.generation_config):
                    model_output = message['choices'][0]['message']['content']                    
                data.update({f'{self.args.model_name}_response': model_output})
                inputs.append({"role":"assistant", "content": model_output})
                if round_num % 2 == 0:
                    input_dict = {
                        'output_a': output_a,
                        'output_b': data[self.args.second_pair_key]
                    }
                else:
                    input_dict = {
                        'output_a': data[self.args.second_pair_key],
                        'output_b': output_a
                    }
                user_input = self.args.user_input.format_map(input_dict)
                inputs.append({"role":"user", "content": user_input})
                model_output = ""
                for message in chose_client.chat_completions_v1(
                    model = self.args.model_class,
                    messages=inputs,
                    stream=False,
                    **self.generation_config):
                    model_output = message['choices'][0]['message']['content']  
                data.update({f'{self.args.model_name}_{round_num}': model_output})
                self.generate_text.put(data, timeout=8)
                break
            except Exception as e:
                print(f"error:{e} with try time {try_times}", flush=True)
                time.sleep(32)
        if data_index % 100 == 0:
            print(f"INDEX:{data_index}\nDATA:{data}", flush=True)

def thread_with_timeout(func, data, num, max_execution_time):
    start_time = time.time()
    def check_time():
        if time.time() - start_time >= max_execution_time:
            print(f"Thread {data['index']} exceeded timeout")
            thread.cancel()

    thread = threading.Thread(target=func, args=(data, num))
    thread.start()

    while thread.is_alive():
        check_time()
        time.sleep(1)


if __name__ == '__main__':
    args = parse_args()
    args.template = args.template.replace('\\n', '\n')
    args.user_input = args.user_input.replace('\\n', '\n')
    data_files = [f for f in os.listdir(args.dataset_dir) if f.endswith('.jsonl')]
    file = sorted(data_files)[args.file_index]
    file_path = osp.join(args.dataset_dir, file)
    print(f"Processing {file_path}")
    random.seed(args.seed)
    queries = [d for d in jsonlines.open(file_path, 'r')]
    if args.num_prompts < len(queries):
        queries = random.sample(queries, args.num_prompts)
    for index, query in enumerate(queries):
        query['index'] = index
    print(f"queries:{queries[0]}")
    server_lists = []
    for i in range(args.client_port, args.client_port+args.num_clients):
        server_lists.append(f"http://0.0.0.0:{i}")
    engine = Engine(args, server_lists)
    engine = Engine(args, server_lists)
    func = None

    if args.task_name == "dialog_inference":
        func = engine.dialog_inference
    elif args.task_name == "direct_feedback":
        func = engine.direct_feedback
    elif args.task_name == "reference_feedback":
        func = engine.reference_feedback
    else:
        raise ValueError("task_name is not supported.")
    
    for num in range(args.round_num):
        print(f"Round {num} start for {len(queries)} queries.")
        with concurrent.futures.ThreadPoolExecutor(args.concurrency) as executor:
            for data in queries:
                executor.submit(thread_with_timeout, func, data, num, 300)
            executor.shutdown()
        queries = sorted(list(engine.generate_text.queue), key=lambda x: x['index'])
        engine.clean_generate_text()
    
    if len(queries) > 0:
        stage2_path = osp.join(args.output_dir, f"{args.model_name}/{file}")
        os.makedirs(os.path.dirname(stage2_path), exist_ok=True)
        writer = jsonlines.open(stage2_path, 'w')
        writer.write_all(queries)
        writer.close()