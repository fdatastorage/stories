# python generate.py dataset_folder target_folder checkpoint_prefix start_checkpoint max_token_length checkpoint_steps skip_first_n

# python generate.py data checkpoints ckp_bdsm hivemind/gpt-j-6B-8bit 2048 300 0

import sys
import transformers

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import datetime
from bitsandbytes.optim import Adam8bit
from import_stuff import GPTJForCausalLM
from datasets import load_dataset
import os

if __name__ == "__main__":
    dataset_folder = sys.argv[1]
    target_folder = sys.argv[2]
    checkpoint_prefix = sys.argv[3]
    start_checkpoint = sys.argv[4]
    max_token_length = int(sys.argv[5])
    checkpoint_steps = int(sys.argv[6])
    skip_first_n = int(sys.argv[7])

    config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    gpt = GPTJForCausalLM.from_pretrained(start_checkpoint, low_cpu_mem_usage=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpt.to(device)

    start_time = datetime.datetime.now()

    i = 0
    gpt.gradient_checkpointing_enable()

    data = load_dataset(dataset_folder, streaming=True)

    optimizer = Adam8bit(gpt.parameters(), lr=1e-5)

    def remove_folder(folder):
      try:
        [os.remove(f"{folder}/{x}") for x in os.listdir(folder)]
        os.rmdir(folder)
      except:
        pass

    def get_number_from_folder(folder_name):
      return int(folder_name.split("-")[-1])

    def remove_folders_except_last_n(n):
      folders = [x for x in os.listdir(target_folder) if x.startswith(checkpoint_prefix)]
      folders = sorted(folders)
      for f in folders[:-n]:
        remove_folder(f"{ target_folder }/{f}")


    with torch.cuda.amp.autocast():
        for row in tqdm(data["train"]):
            if (len(row['0']) <= 1) or (i <= skip_first_n):
              i += 1
              continue

            batch = tokenizer(row['0'], truncation=True, max_length=max_token_length, return_tensors='pt')
            batch = {k: v.cuda() for k, v in batch.items()}

            out = gpt.forward(**batch,)

            loss = F.cross_entropy(out.logits[:, :-1, :].flatten(0, -2), batch['input_ids'][:, 1:].flatten(),
                                  reduction='mean')
            
            i+=1
            if i % checkpoint_steps == 0:
              print("i: ", i)
              print("Loss: ", loss)
              print("Total time in s: ", (datetime.datetime.now() - start_time).total_seconds())
              print("--------")
              remove_folders_except_last_n(1)
              gpt.save_pretrained(f"{target_folder}/{checkpoint_prefix}-{str(i).zfill(6)}")
              print("Model saved.")
              print("=======")
            
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
