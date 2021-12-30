import os
import random
import wandb
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Set, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer

from model import util, generate_attention_mask_from_mask_indicies_and_instruction_tensors
from .generation_metrics import calc_bleu, calc_perplexity, calc_bert_score

EPOCH_ZERO_BERT_SCORES_F1 = {
}

def calculate_t_and_p_value(a, b):
    return stats.ttest_ind(a, b)

def decode_instruction(instrcution_tensor, tokenizer):
    return tokenizer.decode(instrcution_tensor).replace(
                "<|endoftext|>", "").replace("!", "")

def get_sentence_prob_and_len_from_logits(logits, instruction_tensors):
    batch_size, seq_length = logits.shape[0], logits.shape[1]-1
    if len(logits.shape) == 2:
        token_log_probs = logits
        labels = instruction_tensors[..., 1:]
        token_log_probs[labels == 0] = 0 
        sent_probs = torch.exp(torch.sum(token_log_probs, 1))
    else:
        logits = logits[:,:-1,:]
        probs = F.softmax(logits, 2)
        batch_tensor = torch.tensor([i for i in range(batch_size)])
        batch_tensor = batch_tensor.repeat(seq_length, 1)
        batch_tensor = batch_tensor.permute(1, 0)
        seq_tensor = torch.tensor([i for i in range(seq_length)])
        seq_tensor = seq_tensor.repeat(batch_size, 1)
        labels = instruction_tensors[..., 1:]
        token_probs = probs[batch_tensor, seq_tensor, labels]
        token_probs[labels == 0] = 1 
        sent_probs = torch.exp(torch.sum(torch.log(token_probs),1))
    sent_probs = sent_probs.tolist()
    sent_lengths =  -torch.sum(labels == 0, 1) + seq_length  
    sent_lengths = sent_lengths.tolist()
    return sent_probs, sent_lengths

def get_sentence_entrophy(logits, instruction_tensors, tokenizer):
    seq_length = logits.shape[1]-1
    logits = logits[:,:-1,:]
    probs = F.softmax(logits, 2)
    entorphies =  torch.sum(-probs * torch.log(probs), 2)
    labels = instruction_tensors[..., 1:]
    entorphies[labels == 0] = 0 
    sent_lengths =  -torch.sum(labels == 0, 1) + seq_length
    mean_entorphies = torch.sum(entorphies, 1)
    mean_entorphies /= sent_lengths
    return mean_entorphies.tolist()

def get_decoder_outputs_and_metircs(decoder, encoded_prompt, state_embeddings, instruction_tensors, attention_mask, tokenizer, sample_size:int =5, opts: List[str] = ["ground-truth", "argmax", "samples"]):
    """
    Sentence-level preplexity: https://stats.stackexchange.com/questions/129352/how-to-find-the-perplexity-of-a-corpus#:~:text=As%20you%20said%20in%20your,We%20are%20done.&text=And%20this%20is%20the%20perplexity,to%20the%20number%20of%20words.
    """
    batch_size = instruction_tensors.shape[0]
    outputs: List[Dict] = [{} for i in range(batch_size)]

    if "ground-truth" in opts:
        # return perplexity
        gt_output_sequences = decoder(instruction_tensors, condition_embs=state_embeddings, labels=instruction_tensors, attention_mask=attention_mask, validation=True)
        sent_probs, sent_lengths = get_sentence_prob_and_len_from_logits(gt_output_sequences[2], instruction_tensors)
        entrophies = get_sentence_entrophy(gt_output_sequences[2], instruction_tensors, tokenizer)
        gt_instructions = [decode_instruction(seq, tokenizer) for seq in instruction_tensors]
        for i in range(batch_size):
            outputs[i]["ground-truth"] = gt_instructions[i]
            outputs[i]["loss"] = gt_output_sequences[1][i].item()
            outputs[i]["prob"] = sent_probs[i]
            outputs[i]["perplexity"] = calc_perplexity(sent_probs[i], sent_lengths[i])
            outputs[i]["entrophy"] = entrophies[i]
    if "argmax" in opts:
        # return sentence and sentence probability
        pred_argmax_sequences = decoder.generate(
            input_ids=encoded_prompt,
            condition_embs=state_embeddings,
            attention_mask=attention_mask,
            do_sample=False,
            max_length=50,
            temperature=1.0,
            num_beams=5,
            top_k=500,
            top_p=1.0,
            eos_token_ids=[tokenizer.eos_token_id],
            repetition_penalty=1.0,
        )
        argmax_instructions = [decode_instruction(seq, tokenizer) for seq in pred_argmax_sequences[0]]
        sent_probs = torch.exp(pred_argmax_sequences[1])

        Ps, Rs, F1s = calc_bert_score(gt_instructions, argmax_instructions)
        for i in range(batch_size):
            outputs[i]["argmax"] = argmax_instructions[i]
            outputs[i]["argmax-prob"] = sent_probs[i]
            outputs[i]["argmax-bleu"] =  calc_bleu(instruction_tensors[i], pred_argmax_sequences[0][i], tokenizer)
            outputs[i]["argmax-bertScore-P"] = Ps[i]
            outputs[i]["argmax-bertScore-R"] = Rs[i]
            outputs[i]["argmax-bertScore-F1"] = F1s[i]

    if "samples" in opts:
        # return sentence and sentence probability
        for i in range(batch_size):
            outputs[i]["samples"] = []
            outputs[i]["sample-probs"] = []
            outputs[i]["sample-bleus"] = []
            outputs[i]["sample-perplexity"] = []
            outputs[i]["sample-bertScore-P"] = []
            outputs[i]["sample-bertScore-R"] = []
            outputs[i]["sample-bertScore-F1"] = []

        for _ in range(sample_size):
            pred_sampled_sequences = decoder.generate(
                input_ids=encoded_prompt,
                condition_embs=state_embeddings,
                attention_mask=attention_mask,
                do_sample=True,
                max_length=50,
                temperature=0.5,
                num_beams=1,
                top_k=500,
                top_p=1.0,
                eos_token_ids=[tokenizer.eos_token_id],
                repetition_penalty=1.0,
            )
            sent_probs, sent_lengths = get_sentence_prob_and_len_from_logits(pred_sampled_sequences[1], pred_sampled_sequences[0])

            sampled_instructions = [decode_instruction(seq, tokenizer) for seq in pred_sampled_sequences[0]]
            Ps, Rs, F1s = calc_bert_score(gt_instructions, sampled_instructions)

            for i in range(batch_size):
                sampled_sent = sampled_instructions[i]
                sample_bleu = calc_bleu(instruction_tensors[i], pred_sampled_sequences[0][i], tokenizer)
                sample_perplexity = calc_perplexity(sent_probs[i], sent_lengths[i])

                outputs[i]["samples"].append(sampled_sent)
                outputs[i]["sample-probs"].append(sent_probs[i])
                outputs[i]["sample-bleus"].append(sample_bleu)
                outputs[i]["sample-perplexity"].append(sample_perplexity)
                outputs[i]["sample-bertScore-P"].append(Ps[i])
                outputs[i]["sample-bertScore-R"].append(Rs[i])
                outputs[i]["sample-bertScore-F1"].append(F1s[i])
    return outputs

def log_stats(outputs, steps: int, num_display_instructions: int = 250, cala_p_t_sore: bool = False, orig_prefix: str = "", logging_wandb: bool = False):
    # calculate and output statistics to wandb
    metrics = {}
    losses = []
    probs = []
    entrophies = []
    perplexities = []
    argmax_bleus = []
    argmax_probs = []
    argmax_bertScore_Ps = []
    argmax_bertScore_Rs = []
    argmax_bertScore_F1s = []
    sample_bleus = []
    sample_probs = []
    sample_bertScore_Ps = []
    sample_bertScore_Rs = []
    sample_bertScore_F1s = []
    stacked_sample_bertScore_F1s = []

    for op in outputs:
        losses.append(op["loss"])
        probs.append(op["prob"])
        entrophies.append(op["entrophy"])
        perplexities.append(op["perplexity"])
        argmax_bleus.append(op["argmax-bleu"])
        argmax_probs.append(op["argmax-prob"])
        argmax_bertScore_Ps.append(op["argmax-bertScore-P"])
        argmax_bertScore_Rs.append(op["argmax-bertScore-R"])
        argmax_bertScore_F1s.append(op["argmax-bertScore-F1"])
        sample_bleus += op["sample-bleus"]
        sample_probs += op["sample-probs"]
        sample_bertScore_Ps += op["sample-bertScore-P"]
        sample_bertScore_Rs += op["sample-bertScore-R"]
        sample_bertScore_F1s += op["sample-bertScore-F1"]
        stacked_sample_bertScore_F1s.append(op["sample-bertScore-F1"])

    prefix = "Dev_mean/" if orig_prefix == "" else "{}_mean/".format(orig_prefix)
    metrics["{}gt-loss".format(prefix)] = np.mean(losses)
    metrics["{}gt-probs".format(prefix)] = np.mean(probs)
    metrics["{}gt-entrophy".format(prefix)] = np.mean(entrophies)
    metrics["{}gt-perplexity".format(prefix)] = np.mean(perplexities)
    metrics["{}argmax-bleu".format(prefix)] = np.mean(argmax_bleus)
    metrics["{}argmax-probs".format(prefix)] = np.mean(argmax_probs)
    metrics["{}argmax_bertScore_P".format(prefix)] = np.mean(argmax_bertScore_Ps)
    metrics["{}argmax_bertScore_R".format(prefix)] = np.mean(argmax_bertScore_Rs)
    metrics["{}argmax_bertScore_F1".format(prefix)] = np.mean(argmax_bertScore_F1s)
    metrics["{}sample-bleu".format(prefix)] = np.mean(sample_bleus)
    metrics["{}sample-probs".format(prefix)] = np.mean(sample_probs)
    metrics["{}sample-bertScore-P".format(prefix)] = np.mean(sample_bertScore_Ps)
    metrics["{}sample-bertScore-R".format(prefix)] = np.mean(sample_bertScore_Rs)
    metrics["{}sample-bertScore-F1".format(prefix)] = np.mean(sample_bertScore_F1s)

    prefix = "Dev_median/" if orig_prefix == "" else "{}_median/".format(orig_prefix)
    metrics["{}gt-loss".format(prefix)] = np.median(losses)
    metrics["{}gt-probs".format(prefix)] = np.median(probs)
    metrics["{}gt-entrophy".format(prefix)] = np.median(entrophies)
    metrics["{}gt-perplexity".format(prefix)] = np.median(perplexities)
    metrics["{}argmax-bleu".format(prefix)] = np.median(argmax_bleus)
    metrics["{}argmax-probs".format(prefix)] = np.median(argmax_probs)
    metrics["{}argmax_bertScore_P".format(prefix)] = np.median(argmax_bertScore_Ps)
    metrics["{}argmax_bertScore_R".format(prefix)] = np.median(argmax_bertScore_Rs)
    metrics["{}argmax_bertScore_F1".format(prefix)] = np.median(argmax_bertScore_F1s)
    metrics["{}sample-bleu".format(prefix)] = np.median(sample_bleus)
    metrics["{}sample-probs".format(prefix)] = np.median(sample_probs)
    metrics["{}sample-bertScore-P".format(prefix)] = np.median(sample_bertScore_Ps)
    metrics["{}sample-bertScore-R".format(prefix)] = np.median(sample_bertScore_Rs)
    metrics["{}sample-bertScore-F1".format(prefix)] = np.median(sample_bertScore_F1s)


    # calculate p-value
    if cala_p_t_sore:
        stacked_sample_bertScore_F1s = np.stack(stacked_sample_bertScore_F1s,1)

        prefix = "Dev_mean/" if orig_prefix == "" else "{}_mean/".format(orig_prefix)
        stacked_mean_sample_bertScore_F1s = np.mean(stacked_sample_bertScore_F1s, 1)
        if prefix not in EPOCH_ZERO_BERT_SCORES_F1.keys():
            EPOCH_ZERO_BERT_SCORES_F1[prefix] = stacked_mean_sample_bertScore_F1s
            t, p = 0, 1.0
        else:
            t, p = calculate_t_and_p_value(stacked_mean_sample_bertScore_F1s, EPOCH_ZERO_BERT_SCORES_F1[prefix])
        metrics["{}sample-bertScore-F1-t".format(prefix)] = t
        metrics["{}sample-bertScore-F1-p".format(prefix)] = p

        prefix = "Dev_median/" if orig_prefix == "" else "{}_median/".format(orig_prefix)
        stacked_median_sample_bertScore_F1s = np.median(stacked_sample_bertScore_F1s, 1)
        if prefix not in EPOCH_ZERO_BERT_SCORES_F1.keys():
            EPOCH_ZERO_BERT_SCORES_F1[prefix] = stacked_median_sample_bertScore_F1s
            t, p = 0, 1.0
        else:
            t, p = calculate_t_and_p_value(stacked_median_sample_bertScore_F1s, EPOCH_ZERO_BERT_SCORES_F1[prefix])
        metrics["{}sample-bertScore-F1-t".format(prefix)] = t
        metrics["{}sample-bertScore-F1-p".format(prefix)] = p

    if logging_wandb:
        wandb.log(metrics, step=steps, commit=False)
    print(metrics)

    # complie example
    if logging_wandb:
        prefix = "Dev/" if orig_prefix == "" else "{}/".format(orig_prefix)
        random.seed(7777) # fix a seed and reset a random generator to always pick the same examples
        num_display_instructions = len(outputs) if len(outputs) < num_display_instructions else num_display_instructions
        table = wandb.Table(columns=["data_idx", "groundtruth", "arg-max", "sample-1", "sample-2", "sample-3"])
        for op in random.sample(outputs, num_display_instructions):
            table.add_data(op["data_idx"], op["ground-truth"], op["argmax"], op["samples"][0], op["samples"][1], op["samples"][2])

        wandb.log({"{}/instructions".format(prefix): table},
                step=steps, commit=False)

def log_loss_to_txt(all_outputs, file_name: str = "data/analysis_data/loss/2020-10-10-train-1.txt"):
    outfile = open(file_name, "w")
    for output in all_outputs:
        traj_type = os.path.basename(output["data_idx"]).split("_")[-1].replace(".pkl", "")
        data_filename = output["data_idx"].replace("_{}".format(traj_type), "")
        loss = output["loss"]
        outfile.write("{} {} {}\n".format(data_filename, traj_type, loss))
    outfile.write("\n")

def evaluate(encoder, decoder, val_loader: torch.utils.data.DataLoader, tokenizer: GPT2Tokenizer, steps: int, logging_wandb: bool = False, sample_size: int = 5, cala_p_t_sore: bool = False, prefix: str = "", eval_opts: List[str]=None):
    """
    - evaluate on static data split
    - (optional) return eval metics (bleu, gt-perplexity, gt-prob, pred-prob)
    - (optional) return ground-truth and prediction
    """
    all_outputs = []
    for batch in val_loader:
        state_tensors, instruction_tensors, traj_tensors, data_ind, reward, data_label = batch
        state_tensors = state_tensors.to(util.DEVICE)
        instruction_tensors = instruction_tensors.to(util.DEVICE)
        traj_tensors = [traj.to(util.DEVICE) for traj in traj_tensors]
        batch_size = state_tensors.shape[0]

        # embed state / trajectory reps for each example and generate attention masks
        encoder_inputs = (state_tensors, traj_tensors)
        if batch_size == 1:
            state_embeddings = encoder(*encoder_inputs)[0]
            attention_mask = None
        else:
            state_embeddings, feature_attention_mask = encoder(*encoder_inputs)
            attention_mask = generate_attention_mask_from_mask_indicies_and_instruction_tensors(feature_attention_mask, instruction_tensors)

        encoded_prompt = torch.tensor([[tokenizer.bos_token_id] for i in range(batch_size)]).to(util.DEVICE)
        if eval_opts is not None:
            outputs = get_decoder_outputs_and_metircs(decoder, encoded_prompt, state_embeddings, instruction_tensors, attention_mask, tokenizer, sample_size, opts=eval_opts)
        else:
            outputs = get_decoder_outputs_and_metircs(decoder, encoded_prompt, state_embeddings, instruction_tensors, attention_mask, tokenizer, sample_size)

        for i in range(batch_size):
            outputs[i]["data_idx"] = data_ind[i]
        all_outputs += outputs

    log_stats(all_outputs, steps, cala_p_t_sore=cala_p_t_sore, orig_prefix=prefix, logging_wandb=logging_wandb)
