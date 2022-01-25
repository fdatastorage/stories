# python generate.py target_folder file_prefix amount gen_min_len gen_max_len checkpoint_path temperature num_per_one_batch

# python generate.py generated generated 10 1500 2048 hivemind/gpt-j-6B-8bit .9 5

import sys
import json

import datetime
import torch
import transformers

from import_stuff import GPTJForCausalLM

if __name__ == "__main__":
    out_folder = sys.argv[1]
    file_prefix = sys.argv[2]
    story_amt = sys.argv[3]
    min_len = int(sys.argv[4])
    max_len = int(sys.argv[5])
    checkpoint_path = sys.argv[6],
    temperature = float(sys.argv[7])
    batch_size = int(sys.argv[8])

    config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    gpt = GPTJForCausalLM.from_pretrained(checkpoint_path, low_cpu_mem_usage=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpt.to(device)

    def generate_text(prompt_text):
      prompt = tokenizer(prompt_text, return_tensors='pt')
      prompt = {key: value.to(device) for key, value in prompt.items()}
      out = gpt.generate(**prompt, min_length=min_len,
            temperature=temperature, max_length=max_len, do_sample=True, 
            num_return_sequences=batch_size)
      torch.cuda.empty_cache()
      return out
      
    res = []
    while story_amt > 0:
      story_amt -= batch_size
      res += [tokenizer.decode(x)[len('<|startoftext|> '):] for x in generate_text('<|startoftext|> ')]
      while len(res) > 9:
        with open(f"{ out_folder }/{file_prefix}_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S-%f')}.json", encoding='utf8') as f:
          json.dump(res[:10], f, ensure_ascii=False)

        res = res[10:]
