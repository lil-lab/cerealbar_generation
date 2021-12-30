# Script to train models

import copy, os, sys, time, pickle
import logging
import numpy as np
from IPython import embed
import wandb #! importing wandb first breaks IPython import
from argparse import ArgumentParser, Namespace
from contextlib import nullcontext
from typing import List, Dict

import torch
import torch.nn.functional as F
import torch.multiprocessing
from torch.nn.utils.rnn import pad_sequence
torch.multiprocessing.set_sharing_strategy('file_system')
from transformers import GPT2Tokenizer, GPT2Config, AdamW

from model import CNNLSTMStateEncodingModel, GPT2PSALMHeadModel, generate_attention_mask_from_mask_indicies_and_instruction_tensors, util
from learning.utils import *
from learning.dataset import INSTRUCTION_DATASET, collate_fn
from learning.args import TrainingArgs
from learning.evaluation import evaluate

KL_DIV = torch.nn.KLDivLoss()

def get_log_tensor_cap_values(cap_vals: Dict):
    if "token_min_val" in cap_vals:
        token_min_val = float(cap_vals["token_min_val"])
        token_min_val = torch.log(torch.tensor([token_min_val])).item()
    else:
        token_min_val = None
    if "token_max_val" in cap_vals:
        token_max_val = float(cap_vals["token_max_val"])
        token_max_val = torch.log(torch.tensor([token_max_val])).item()
    else:
        token_max_val = None
    if "sent_min_val" in cap_vals:
        sent_min_val = float(cap_vals["sent_min_val"])
        sent_min_val = torch.log(torch.tensor([sent_min_val])).item()
    else:
        sent_min_val = None
    if "sent_max_val" in cap_vals:
        sent_max_val = float(cap_vals["sent_max_val"])
        sent_max_val = torch.log(torch.tensor([sent_max_val])).item()
    else:
        sent_max_val = None
    return token_min_val, token_max_val, sent_min_val, sent_max_val

# ! duplicate with get_sentence_prob_and_len_from_logits
def get_sent_log_probs(instruction_tensors: torch.tensor, logits: torch.tensor, pad_token_id: int = 0, normalize_by_length: bool=False, cap_vals: Dict = None) -> torch.tensor:
    sent_log_probs = []
    shift_probs = F.log_softmax(logits[..., :-1, :].contiguous(), 2)
    shift_labels = instruction_tensors[..., 1:].contiguous()
    batch_size = shift_labels.shape[0]

    # probability cap values
    if cap_vals is not None:
        token_min_val, token_max_val, sent_min_val, sent_max_val = get_log_tensor_cap_values(cap_vals)
    else:
        token_min_val, token_max_val, sent_min_val, sent_max_val = None, None, None, None

    for i in range(batch_size):
        sub = torch.sum(shift_labels[i,...] == pad_token_id)
        seq_len = shift_labels.shape[1]
        instance_shift_probs = shift_probs[i, :seq_len-sub, :]
        instance_shift_labels = shift_labels[i, :seq_len-sub]
        sequence_tensors = torch.tensor([j for j in range(seq_len-sub)])
        token_log_probs = shift_probs[i, ...][sequence_tensors, instance_shift_labels]

        # cap token_log_probs
        if token_min_val is not None and token_max_val is not None:
            token_log_probs = torch.clamp(token_log_probs, min=token_min_val, max=token_max_val)
        elif token_min_val is not None:
            token_log_probs = torch.clamp(token_log_probs, min=token_min_val)
        elif token_max_val is not None:
            token_log_probs = torch.clamp(token_log_probs, max=token_max_val)

        # calculae sent-level log probablities
        # ! todo: check the basis of this normalization term
        if normalize_by_length:
            sent_log_prob = torch.mean(token_log_probs)
        else:
            sent_log_prob = torch.sum(token_log_probs)

        # cap sent_log_probs
        if sent_min_val is not None and sent_max_val is not None:
            sent_log_prob = torch.clamp(sent_log_prob, min=sent_min_val, max=sent_max_val)
        elif sent_min_val is not None:
            sent_log_prob = torch.clamp(sent_log_prob, min=sent_min_val)
        elif sent_max_val is not None:
            sent_log_prob = torch.clamp(sent_log_prob, max=sent_max_val)

        sent_log_probs.append(sent_log_prob)
    return torch.stack(sent_log_probs)

def get_token_log_probs(instruction_tensors: torch.tensor, logits: torch.tensor, pad_token_id: int = 0) -> List[torch.tensor]:
    token_log_probs = []
    shift_probs = F.log_softmax(logits[..., :-1, :].contiguous(), 2)
    shift_labels = instruction_tensors[..., 1:].contiguous()
    batch_size = shift_labels.shape[0]
    # todo: this will break if "!" is includes in a ground-truth seueunce
    for i in range(batch_size):
        sub = torch.sum(shift_labels[i,...] == pad_token_id)
        seq_len = shift_labels.shape[1]
        instance_shift_probs = shift_probs[i, :seq_len-sub, :]
        instance_shift_labels = shift_labels[i, :seq_len-sub]
        sequence_tensors = torch.tensor([j for j in range(seq_len-sub)])
        token_log_prob = shift_probs[i, ...][sequence_tensors, instance_shift_labels]
        token_log_probs.append(token_log_prob)
    return token_log_probs

def get_reward_weights(current_encoder, current_decoder, batch, sent_probs, reward_weighting: str = "", weight_type: str = ""):
    # todo: clean duplicate code
    # todo: remove token-level weights
    """
    Calculating reward reweighting term
    IPS, IPS + R
    References: Lawrence et al., 2017 (https://arxiv.org/pdf/1707.09118.pdf), Kreutzer et al., (https://arxiv.org/pdf/2011.02511.pdf)
    """
    token_level = reward_weighting["token_level"] if "token_level" in reward_weighting else False
    method = reward_weighting["method"] if "method" in reward_weighting else False
    min_clamp_value = reward_weighting["min_clamp_value"] if "min_clamp_value" in reward_weighting else None
    max_clamp_value = reward_weighting["max_clamp_value"] if "max_clamp_value" in reward_weighting else None

    # putting models in evaluation mode
    current_encoder.train()
    current_encoder._lstm.dropout = 0.0
    current_decoder.eval()

    # calucalte wwights
    reward_weights = None
    if batch is None:
        return reward_weights

    batch_size = batch[0].shape[0]

    if method is not None and method != "":
        # forward pass on s; examples
        if batch is not None:
            with torch.no_grad():
                state_tensors, instruction_tensors, traj_tensors, file_names, rewards, _ = batch
                if weight_type == "negative_only":
                    reward_weights = torch.ones(state_tensors.shape[0]).to(util.DEVICE)
                    weight_index = rewards < 0
                    if torch.all(weight_index == False):
                        return reward_weights
                    state_tensors = state_tensors[weight_index]
                    instruction_tensors = instruction_tensors[weight_index]
                    file_names = [f for i, f in enumerate(file_names) if weight_index[i]]
                    traj_tensors = [t for i, t in enumerate(traj_tensors) if weight_index[i]]
                state_tensors = state_tensors.to(util.DEVICE)
                instruction_tensors = instruction_tensors.to(util.DEVICE)
                traj_tensors = [traj.to(util.DEVICE) for traj in traj_tensors]
                batch_size = state_tensors.shape[0]
                encoder_inputs = (state_tensors, traj_tensors)
                
                state_embeddings, attention_mask = encoder_step(current_encoder, encoder_inputs, instruction_tensors)
                current_decoder_outputs = current_decoder(instruction_tensors, condition_embs=state_embeddings,
                                labels=instruction_tensors, attention_mask=attention_mask, validation=True)
                current_sent_log_probs = get_sent_log_probs(instruction_tensors, current_decoder_outputs[2], normalize_by_length=False)
                current_sent_probs = torch.exp(current_sent_log_probs)
                # ! so far ips weighting only supports negative examples
                init_sent_probs = torch.ones(current_sent_probs.shape).to(util.DEVICE)
                for i, f in enumerate(file_names):
                    if f in sent_probs:
                        init_sent_probs[i] = sent_probs[f]

                if weight_type == "negative_only":
                    reward_weights[weight_index] = current_sent_probs / init_sent_probs
                else:
                    reward_weights = current_sent_probs / init_sent_probs
                    reward_weights = reward_weights.detach()

    # clamp reward weights
    if min_clamp_value is not None:
        if reward_weights is not None:
            reward_weights = torch.clamp(reward_weights, min_clamp_value, max_clamp_value)

    return reward_weights

def encoder_step(encoder, encoder_inputs, instruction_tensors):
    batch_size = instruction_tensors.shape[0]
    if batch_size == 1:
        state_embeddings = encoder(*encoder_inputs)[0]
        attention_mask = None
    else:
        state_embeddings, feature_attention_mask = encoder(*encoder_inputs)
        attention_mask = generate_attention_mask_from_mask_indicies_and_instruction_tensors(feature_attention_mask, instruction_tensors)

    return state_embeddings, attention_mask

def bandit_step(encoder, decoder, optimizer, batch, tokenizer, logger, steps, use_dropout: bool, norm_type: str, reward_weights: torch.tensor = None, weight_method: str = '', cap_vals: Dict = None, stats_prefix: str="bandit"):
    if use_dropout:
        encoder.train()
        encoder._lstm.dropout = 0.5
        decoder.train()
    else:
        encoder.train()
        encoder._lstm.dropout = 0.0
        decoder.eval()

    # put data on device
    state_tensors, instruction_tensors, traj_tensors, file_names, rewards, data_label = batch
    state_tensors = state_tensors.to(util.DEVICE)
    instruction_tensors = instruction_tensors.to(util.DEVICE)
    traj_tensors = [traj.to(util.DEVICE) for traj in traj_tensors]
    rewards = rewards.to(util.DEVICE)
    batch_size = state_tensors.shape[0]
    encoder_inputs = (state_tensors, traj_tensors)

    # training stats
    train_stats = {
    }
    sent_log_probs = None

    # process reward weights
    if reward_weights is None:
        reward_weights = torch.ones([batch_size]).to(util.DEVICE)

    # forward pass (encoder)
    state_embeddings, attention_mask = encoder_step(encoder, encoder_inputs, instruction_tensors)

    # forward pass (decoder)
    # todo: make sure the output value is the same with torch.mean and reduction='mean'
    reduction = 'none' if norm_type == "gpt2" else 'mean'
    decoder_outputs = decoder(instruction_tensors, condition_embs=state_embeddings,
                    labels=instruction_tensors, attention_mask=attention_mask, validation=True, reduction=reduction)

    # calculate loglikelihood
    if norm_type == "gpt2":
        neg_loglikelihood = decoder_outputs[0]
    else:
        if norm_type == "vanila-pg":
            # todo: do this
            sent_log_probs = get_sent_log_probs(instruction_tensors, decoder_outputs[2], normalize_by_length=False, cap_vals=cap_vals)
            neg_loglikelihood = -sent_log_probs
        elif norm_type == "normed-pg":
            sent_log_normalized_probs = get_sent_log_probs(instruction_tensors, decoder_outputs[2], normalize_by_length=True, cap_vals=cap_vals)
            sent_log_normalized_probs_list = torch.exp(sent_log_normalized_probs).tolist()
            neg_loglikelihood = -sent_log_normalized_probs
            train_stats["{}_normalized_sent_probs".format(stats_prefix)] = sent_log_normalized_probs_list
            for i in range(len(sent_log_normalized_probs_list)):
                name = "{}_normalized_sent_probs".format(data_label[i])
                train_stats.setdefault(name, [])
                train_stats[name].append(sent_log_normalized_probs_list[i])
        else:
            raise ValueError("norm_type {} is not supported.".format(norm_type))

    # calculate rewards and loss
    if weight_method == "ips+r":
        assert(norm_type == "vanila-pg")
        if torch.mean(reward_weights) == 0.:
            reward_weights += 1.
        else:
            reward_weights = reward_weights / torch.mean(reward_weights)
        losses = rewards * reward_weights * (neg_loglikelihood - torch.sum(reward_weights * neg_loglikelihood))
        rewards = rewards * reward_weights
    else:
        rewards = rewards * reward_weights
        if norm_type == "gpt2":
            loss_weight = rewards.unsqueeze(1)
            loss_weight = loss_weight.repeat(1, instruction_tensors.shape[-1]-1)
            loss_weight = loss_weight.view(-1) # todo: make sure this is aligned with .view in forward function
        else:
            loss_weight = rewards
        losses = loss_weight * neg_loglikelihood
    loss = torch.mean(losses)
        
    # update stats
    reward_weights_list = reward_weights.tolist()
    rewards_list = rewards.tolist()
    train_stats["{}_reward_weights".format(stats_prefix)] = reward_weights_list
    train_stats["{}_rewards".format(stats_prefix)] = rewards_list
    for i in range(len(reward_weights_list)):
        name = "{}_reward_weights".format(data_label[i])
        train_stats.setdefault(name, [])
        train_stats[name].append(reward_weights_list[i])
        name = "{}_rewards".format(data_label[i])
        train_stats.setdefault(name, [])
        train_stats[name].append(rewards_list[i])
    with torch.no_grad():
        if sent_log_probs is None:
            sent_log_probs = get_sent_log_probs(instruction_tensors, decoder_outputs[2], normalize_by_length=False)
    sent_log_probs_list = torch.exp(sent_log_probs).tolist()
    train_stats["{}_sent_probs".format(stats_prefix)] = sent_log_probs_list

    with torch.no_grad():
        idx = torch.tensor([i for i in range(len(sent_log_probs_list))]).to(util.DEVICE)
        if norm_type == "gpt2":
            idx = idx.unsqueeze(1)
            idx = idx.repeat(1, instruction_tensors.shape[-1]-1)
            idx = idx.view(-1)

        for i in range(len(sent_log_probs_list)):
            # update sent_log_probs for each examples
            name = "{}_sent_probs".format(data_label[i])
            train_stats.setdefault(name, [])
            train_stats[name].append(sent_log_probs_list[i])
            # update loss for each examples
            if norm_type == "gpt2":
                name = "{}_loss".format(data_label[i])
                train_stats.setdefault(name, [])
                train_stats[name] += losses[idx == i].tolist()
            else:
                name = "{}_loss".format(data_label[i])
                train_stats.setdefault(name, [])
                train_stats[name].append(losses[i].tolist())
    return loss, train_stats

def calc_kl_loss(current_decoder_outputs: torch.tensor, init_decoder_outputs: torch.tensor):
    # todo: potentially remove padding tokens
    # todo: current_probs vs init_probs (whiich one should come first)??
    # todo summation ???
    # ! this is including padding tokens
    init_probs = F.log_softmax(init_decoder_outputs[..., :-1, :], 2)
    init_probs = init_probs.permute(2,0,1)
    current_probs = F.softmax(current_decoder_outputs[..., :-1, :], 2)
    current_probs = current_probs.permute(2,0,1)
    # ? reference: https://www.programcreek.com/python/example/104436/torch.nn.functional.kl_div
    # ? https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#kl_div
    # ? https://github.com/pytorch/pytorch/blob/7cc029cb75c292e93d168e117e46a681ace02e79/aten/src/ATen/native/Loss.cpp#L69-L94
    # ? reference (numerical instability): https://github.com/pytorch/pytorch/issues/32520
    # todo: alternative here? (mode-finiding vs ....)
    """ naive implementation of KL
    kl_naive = torch.sum(F.softmax(current_decoder_outputs[..., :-1, :], dim=2) * (F.log_softmax(current_decoder_outputs[..., :-1, :], dim=2) - F.log_softmax(init_decoder_outputs[..., :-1, :], dim=2)),2).mean()
    """
    loss = torch.sum(F.kl_div(init_probs, current_probs, reduction='none'),0).mean()
    return loss

def kl_step(current_encoder, current_decoder, init_encoder, init_decoder, optimizer, batch, tokenizer, logger, steps, dropout_opts: Dict):
    if dropout_opts["init_encoder_dropout"]:
        init_encoder.train()
        init_encoder._lstm.dropout = 0.5
    else:
        init_encoder.train()
        init_encoder._lstm.dropout = 0.0

    if  dropout_opts["init_decoder_dropout"]:
        init_decoder.train()
    else:
        init_decoder.eval()

    if dropout_opts["current_encoder_dropout"]:
        current_encoder.train()
        current_encoder._lstm.dropout = 0.5
    else:
        current_encoder.train()
        current_encoder._lstm.dropout = 0.0

    if dropout_opts["current_decoder_dropout"]:
        current_decoder.train()
    else:
        current_decoder.eval()

    # put data on device
    state_tensors, instruction_tensors, traj_tensors, file_names, rewards, _ = batch

    state_tensors = state_tensors.to(util.DEVICE)
    instruction_tensors = instruction_tensors.to(util.DEVICE)
    traj_tensors = [traj.to(util.DEVICE) for traj in traj_tensors]
    batch_size = state_tensors.shape[0]
    encoder_inputs = (state_tensors, traj_tensors)

    # forward pass on the current model
    state_embeddings, attention_mask = encoder_step(current_encoder, encoder_inputs, instruction_tensors)
    current_decoder_outputs = current_decoder(instruction_tensors, condition_embs=state_embeddings,
                    labels=instruction_tensors, attention_mask=attention_mask, validation=True)

    # forward pass on the initial model
    with torch.no_grad():
        state_embeddings, attention_mask = encoder_step(init_encoder, encoder_inputs, instruction_tensors)
        init_decoder_outputs = init_decoder(instruction_tensors, condition_embs=state_embeddings,
                        labels=instruction_tensors, attention_mask=attention_mask, validation=True)
    # calculate kl loss
    loss = calc_kl_loss(current_decoder_outputs[2], init_decoder_outputs[2])

    # process train_stats
    train_stats = {
        "kl_sent_probs": [],
    }
    sent_log_probs = get_sent_log_probs(instruction_tensors, current_decoder_outputs[2], normalize_by_length=False)
    train_stats["kl_sent_probs"] += torch.exp(sent_log_probs).tolist()

    return  loss, train_stats

def train_loop(encoder, decoder, optimizer, tokenizer, data_loaders: Dict, training_arguments, logger=None, steps=0):
    train_configs = training_arguments.get_config()
    main_loader = iter(data_loaders["main_loader"]) if data_loaders["main_loader"] is not None else None
    aux_loader =  iter(data_loaders["aux_loader"]) if data_loaders["aux_loader"] is not None else None
    kl_loader =  iter(data_loaders["kl_loader"]) if data_loaders["kl_loader"] is not None else None
    full_val_loader = data_loaders["full_val_loader"]
    cleaned_val_loader = data_loaders["cleaned_val_loader"]

    main_length = len(data_loaders["main_loader"]) if data_loaders["main_loader"] is not None else 0
    aux_length =  len(data_loaders["aux_loader"]) if data_loaders["aux_loader"] is not None else 0
    kl_length =  len(data_loaders["kl_loader"]) if data_loaders["kl_loader"] is not None else 0
    max_iterations: int = training_arguments.get_max_epochs() * main_length if "max_iterations" not in train_configs else train_configs["max_iterations"] 

    st = time.time()
    step_cts = {
        "main_ct": 0,
        "aux_ct": 0,
        "kl_ct": 0,
    }
    last_checkpoint_step = 0
    train_stats = {}

    # AMP (https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/)
    use_amp = train_configs["use_amp"] if "use_amp" in train_configs else False
    train_ctx = torch.cuda.amp.autocast() if use_amp else nullcontext()
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # Load intial encoder / decoder for KL step and calculate IPS reward weights
    if "initial_checkpoint_path" in train_configs:
        init_encoder, init_decoder = get_model(train_configs["initial_checkpoint_path"], training_arguments, len(tokenizer.decoder), logger)
        init_encoder.eval()
        init_decoder.eval()
    
    for _ in range(max_iterations):
        optimizer.zero_grad()
        
        # multi-objective loss
        main_loss, aux_loss, kl_loss = torch.tensor([0.]).to(util.DEVICE), torch.tensor([0.]).to(util.DEVICE),  torch.tensor([0.]).to(util.DEVICE)
        total_stats = {}

        # get batch data
        if main_loader is not None:
            if step_cts["main_ct"] >= len(main_loader):
                main_loader = iter(data_loaders["main_loader"])
                step_cts["main_ct"] = 0
            main_batch = next(main_loader)
        else:
            main_batch = None

        if aux_loader is not None:
            if step_cts["aux_ct"] >= len(aux_loader):
                aux_loader =  iter(data_loaders["aux_loader"])
                step_cts["aux_ct"] = 0
            aux_batch = next(aux_loader)
        else:
            aux_batch = None

        if kl_loader is not None:
            if step_cts["kl_ct"] >= len(kl_loader):
                kl_loader =  iter(data_loaders["kl_loader"])
                step_cts["kl_ct"] = 0
            kl_batch = next(kl_loader)
        else:
            kl_batch = None
        
        # get reward weights
        if train_configs["loss_configs"]["reward_weighting"]["weight_method"] != "":
            main_reward_weights = get_reward_weights(encoder, decoder, main_batch, data_loaders["sent_probs"], train_configs["loss_configs"]["reward_weighting"], weight_type=train_configs["loss_configs"]["reward_weighting"]["weight_type"])
            aux_reward_weights = get_reward_weights(encoder, decoder, aux_batch, data_loaders["sent_probs"], train_configs["loss_configs"]["reward_weighting"], weight_type=train_configs["loss_configs"]["reward_weighting"]["weight_type"])
        else:
            main_reward_weights, aux_reward_weights = None, None
        
        with train_ctx:
            # 1. main loss
            if main_batch is not None:
                main_loss, main_train_stats = bandit_step(encoder, decoder, optimizer, main_batch, tokenizer, logger, steps, use_dropout=train_configs["loss_configs"]["main_dropout"], norm_type=train_configs["loss_configs"]["norm_type"], reward_weights=main_reward_weights, weight_method=train_configs["loss_configs"]["reward_weighting"]["weight_method"], stats_prefix="main")
                step_cts["main_ct"] += 1
                total_stats.update(main_train_stats)

            # 2. (separate) bandit loss
            if aux_batch is not None:
                aux_loss, aux_train_stats = bandit_step(encoder, decoder, optimizer, aux_batch, tokenizer, logger, steps, use_dropout=train_configs["loss_configs"]["aux_dropout"], norm_type=train_configs["loss_configs"]["norm_type"], reward_weights=aux_reward_weights, weight_method=train_configs["loss_configs"]["reward_weighting"]["weight_method"], stats_prefix="aux")
                step_cts["aux_ct"] += 1
                total_stats.update(aux_train_stats)

            # 3. kl loss
            if kl_batch is not None:
                kl_loss, kl_train_stats = kl_step(encoder, decoder, init_encoder, init_decoder, optimizer, kl_batch, tokenizer, logger, steps, dropout_opts=train_configs["loss_configs"]["kl_dropout"])
                step_cts["kl_ct"] += 1
                total_stats.update(kl_train_stats)

        main_loss = float(train_configs["loss_configs"]["main_alpha"]) * main_loss
        aux_loss = float(train_configs["loss_configs"]["aux_alpha"]) * aux_loss
        kl_loss = float(train_configs["loss_configs"]["kl_alpha"]) * kl_loss
        total_loss = main_loss + aux_loss + kl_loss

        # backward pass and paramter update
        if use_amp:
            scaler.scale(total_loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            # https://pytorch.org/docs/stable/notes/amp_examples.html
            scaler.unscale_(optimizer) 
        else:
            total_loss.backward()

        # References
        # https://discuss.pytorch.org/t/proper-way-to-do-gradient-clipping/191/14
        if "gradient_clipping" in train_configs.keys():
            if train_configs["gradient_clipping"]["method"] == "clip":
                # this norm strategy is sensitive to # of parameters
                torch.nn.utils.clip_grad_norm((list(encoder.parameters()) + list(decoder.parameters())), float(train_configs["gradient_clipping"]["max_value"]))
            elif train_configs["gradient_clipping"]["method"] == "clamp":
                # https://github.com/pytorch/pytorch/issues/4829
                # https://discuss.pytorch.org/t/gradient-clipping/2836
                for p in encoder.parameters():
                    if p.grad is not None:
                        p.grad.data.clamp_(-float(train_configs["gradient_clipping"]["max_value"]), float(train_configs["gradient_clipping"]["max_value"]))
                for p in decoder.parameters():
                    if p.grad is not None:
                        p.grad.data.clamp_(-float(train_configs["gradient_clipping"]["max_value"]), float(train_configs["gradient_clipping"]["max_value"]))
            else:
                raise ValueError("not supported norm type.")

        if use_amp:
            # Unscales gradients and calls
            # or skips optimizer.step()
            scaler.step(optimizer)
            # Updates the scale for next iteration
            scaler.update()
        else:
            optimizer.step()

        # logging
        train_stats.setdefault("total_loss", [])
        train_stats.setdefault("main_loss", [])
        train_stats.setdefault("aux_loss", [])
        train_stats.setdefault("kl_loss", [])

        train_stats["total_loss"].append(total_loss.item())
        train_stats["main_loss"].append(main_loss.item())
        train_stats["aux_loss"].append(aux_loss.item())
        train_stats["kl_loss"].append(kl_loss.item())

        for key in total_stats.keys():
            train_stats.setdefault(key, [])
            train_stats[key] += total_stats[key]

        if steps % training_arguments.get_logging_step() == (training_arguments.get_logging_step()-1):
            avg_loss = np.mean(train_stats["total_loss"])
            msg = "# Steps {}, {} s: AVG_LOSS {}".format(steps, time.time() - st, np.round(avg_loss, 6))
            print_and_log(msg, logger)
            for key in train_stats.keys():
                wandb.log({"Train_mean/{}".format(key): np.mean(train_stats[key])}, step=steps, commit=False)
            for key in train_stats.keys():
                wandb.log({"Train_median/{}".format(key): np.median(train_stats[key])}, step=steps, commit=False)
            train_stats = {}
            st = time.time()

        # validation
        if int((steps-last_checkpoint_step) / training_arguments.get_checkpoint_step()):
            validate_and_save_model(encoder, decoder, optimizer, full_val_loader, cleaned_val_loader, tokenizer, steps, training_arguments, logger)
            last_checkpoint_step = steps

        # update step
        steps += 1

    # save and validate model at the end of iterations.
    validate_and_save_model(encoder, decoder, optimizer, full_val_loader, cleaned_val_loader, tokenizer, steps, training_arguments, logger)
    
def save_models(encoder, decoder, optimizer, step: int, training_arguments):
    """
    save model and optimizer
    """
    checkpoint = {
        'step': step,
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, "{}/checkpoint_{}.pth".format(training_arguments.get_checkpoint_directory(), step))

def validate(encoder, decoder, val_loader: torch.utils.data.DataLoader, tokenizer: GPT2Tokenizer, logger, steps, sample_size: int = 5, cala_p_t_sore=False, prefix: str = "", eval_opts: List[str]=None):
    """
    This function evaluate models for human-human interaction data for generation metrics (e.g., BLEU).
    We conduct this evaluation to get a coarse sense of whther training is working fine, but we do not rely these metrics for model checkpoint selection.
    """
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        evaluate(encoder, decoder, val_loader, tokenizer, steps, logging_wandb=True, sample_size=sample_size, cala_p_t_sore=cala_p_t_sore, prefix=prefix, eval_opts=eval_opts)

def validate_and_save_model(encoder, decoder, optimizer, full_val_loader, cleaned_val_loader, tokenizer, steps: int, training_arguments, logger = None, cala_p_t_sore=True):
    train_configs = training_arguments.get_config()
    eval_opts = train_configs["eval_opts"] if "eval_opts" in train_configs.keys() else None

    # saving a model
    if training_arguments.get_save_checkpoint():
        save_models(encoder, decoder, optimizer, steps, training_arguments)

    # validating on human-human interaction data
    if cleaned_val_loader is not None:
        # validate on smaller validation data
        # We pair system generated shortest pathes (instead of human user execution) and human generated instructions in this smaller validation data.
        # To generate this dataset, we hand-picked 50~ human generated instructions from the full validation set, in which the shortest pathes generated by our system planner aligns well with the instructions.
        validate(encoder, decoder, cleaned_val_loader, tokenizer, logger, steps, sample_size=10, cala_p_t_sore=cala_p_t_sore, prefix="Cleaned-dev", eval_opts=eval_opts)

    if full_val_loader is not None:
        # validate on full human-human interaction validation data
        validate(encoder, decoder, full_val_loader, tokenizer, logger, steps, sample_size=3, cala_p_t_sore=cala_p_t_sore, prefix="Full-dev", eval_opts=eval_opts)



def get_dataset_loaders(logger, training_arguments, train_configs: Dict, tokenizer: GPT2Tokenizer):
    reward_fnc_dispatcher = {
        "positive_one": positive_one,
        "negative_one": negative_one,
    }

    reward_scalaing_dispatcher = {
        "clamp": clamping,
        "linear": linear_scaling,
    }

    data_pathes = {
        "main_data_path": {},
        "aux_data_path": {},
        "kl_data_path": {},
        "full_val_data_path": {},
        "cleaned_val_data_path": {},
    }

    sent_prob_dicts = {}
    for dp in data_pathes:
        if dp in train_configs.keys():
            data_set = INSTRUCTION_DATASET(tokenizer, logger)
            for cfg in train_configs[dp]:
                data_path = cfg["data_path"]
                reward_fnc = cfg["reward_fnc"] if "reward_fnc" in cfg else None
                if reward_fnc is not None:
                        reward_fnc=reward_fnc_dispatcher[reward_fnc]
                if ".pkl" in data_path:
                    # loading human-human  (suhr et al.,) training data
                    game_id_files = cfg["game_id_files"] if "game_id_files" in cfg else None
                    data_set.add_data_from_suhr_etal(data_path, reward_fnc=reward_fnc, data_label=cfg["label"],  max_examples=int(cfg["max_examples"]), max_games=int(cfg["max_games"]), game_id_files=game_id_files)
                elif ".txt" in data_path:
                    # loading human-system interaction data
                    data_set.add_data_from_human_feedback(data_path, reward_fnc=reward_fnc, data_label=cfg["label"], max_examples=int(cfg["max_examples"]))
                if "sent_prob_path" in cfg:
                    sent_prob_dicts.update(pickle.load(open(cfg["sent_prob_path"], "rb")))

            shuffle = False if "val" in dp else True
            batch_size = train_configs["val_batch_size"] if "val" in dp else train_configs["train_batch_size"] 
            data_loader = torch.utils.data.DataLoader(data_set,
                                                    batch_size=batch_size,
                                                    shuffle=shuffle,
                                                    pin_memory=False,
                                                    num_workers=training_arguments.get_num_workers(),
                                                    collate_fn=collate_fn)
        else:
            data_set = None
            data_loader = None
        data_pathes[dp]["data_set"] = data_set
        data_pathes[dp]["data_loader"] = data_loader
    
    if logger is not None:
        msg = "Dataset created"
        if  data_pathes["main_data_path"]["data_set"] is not None:
            msg += " {} main set".format(len(data_pathes["main_data_path"]["data_set"]))
        if  data_pathes["aux_data_path"]["data_set"] is not None:
            msg += " {} aux set".format(len(data_pathes["aux_data_path"]["data_set"]))
        if data_pathes["kl_data_path"]["data_set"] is not None:
            msg += " {} kl set".format(len(data_pathes["kl_data_path"]["data_set"]))
        if data_pathes["full_val_data_path"]["data_set"] is not None:
            msg += " {} full validation".format(len(data_pathes["full_val_data_path"]["data_set"]))
        if data_pathes["cleaned_val_data_path"]["data_set"] is not None:
            msg += " {} cleaned validation".format(len(data_pathes["cleaned_val_data_path"]["data_set"]))
        print_and_log(msg, logger)

    if "use_kl" not in train_configs["loss_configs"]:
        train_configs["loss_configs"]["use_kl"] = False

    data_loaders = {
        "main_loader": data_pathes["main_data_path"]["data_loader"],
        "aux_loader": data_pathes["aux_data_path"]["data_loader"],
        "kl_loader": data_pathes["kl_data_path"]["data_loader"],
        "full_val_loader": data_pathes["full_val_data_path"]["data_loader"],
        "cleaned_val_loader": data_pathes["cleaned_val_data_path"]["data_loader"],
        "sent_probs": sent_prob_dicts,
    }
    return data_loaders

def get_logger(training_arguments):
    """
    get a logger
    """
    logger_file_name = os.path.join(training_arguments.get_logger_directory(), "logger.log")
    logging.basicConfig(filename=logger_file_name,
                        format='%(asctime)s %(message)s',
                        level=os.environ.get("LOGLEVEL", "INFO"),
                        filemode='w')
    logger = logging.getLogger()
    return logger

def get_wandb(training_arguments):
    """
    set-up wandb
    """
    if training_arguments.get_turnoff_wandb():
        os.system("wandb off")
    else:
        os.system("wandb on")

    # TODO: make sure to change wandb configs for logging to wandb
    if training_arguments.get_debug():
        project_name = "p-instr-gen-hil-learning-debug"
    else:
        project_name = "p-instr-gen-hil-learning"
    if training_arguments.get_resume_training():
        wandb.init(id=training_arguments.get_wandb_id(), project=project_name, config=training_arguments.get_config(),
                   entity="lil", name=training_arguments.get_overwrite_wandb_name())
    else:
        wandb.init(project=project_name, config=training_arguments.get_config(),
                   entity="lil", name=training_arguments.get_wandb_name())

def get_tokenizer() -> GPT2Tokenizer:
    return GPT2Tokenizer.from_pretrained("gpt2")

def get_model(checkpoint_path: str, training_arguments, vocab_size: int, logger, get_only_models: bool= True):
    """
    load a model
    """
    print_and_log("Loading models ...", logger)
    train_configs = training_arguments.get_config()
    if checkpoint_path != '':
        checkpoints = torch.load(checkpoint_path)
        print_and_log("Loading a model from {}".format(checkpoint_path), logger)
    else:
        checkpoints = None

    encoder_configs = {
        "encoder_type": "cnn_lstm",
        "d_embed": 48,
        "d_model": 512,
        "n_depth": 2,
        "rcpf_size": 1,
        "cnn_hex": True,
        "cnn_actv_func": "tanh",
        "cnn_use_norm": True,
        "embeddings_type": "learned",
        "breakpoint_type": "",
        "feature_map_size": 5,
        "feature_filter_size": 3,
        "rotate_feature_map": True,
        "feature_merge_type": "cat",
        "feature_output_dimension": 1400,
        "feature_cnn_n_depth": 1,
        "feature_cnn_actv_func": "tanh",
        "feature_cnn_use_norm": True,
        "lstm_input_merge_type": "cat",
        "lstm_output_merge_type": "spatial-cat",
        "lstm_skip": True,
        "lstm_pb_dim": 48,
        "reduce_feature": False,
        "lstm_d_model": 128,
        "lstm_num_layers": 2,
        "lstm_bidirectional": True,
        "lstm_dropout": 0.5
    }
    encoder = CNNLSTMStateEncodingModel(encoder_configs)
    if checkpoints is not None:
        encoder.load_state_dict(checkpoints["encoder"])

    decoder_config = GPT2Config.from_pretrained("gpt2")
    decoder_config.n_ctx = 2500  # todo: comment out if breaks
    decoder_config.c_embd = encoder.get_dimension()
    decoder_config.vocab_size = vocab_size
    decoder_config.condition_norm = True
    decoder_config.illegal_token_ids = None
    decoder_config.pad_token_id = 0
    decoder_config.length_penalty = 1.0
    decoder_config.num_return_sequences = 1
    decoder = GPT2PSALMHeadModel(decoder_config)
    
    if checkpoints is None:
        decoder_config.n_layer = 10
        decoder = GPT2PSALMHeadModel(decoder_config)
        decoder.load_model("checkpoints/gpt2")
        print_and_log("Loading GPT-2 weights for decoder", logger)
        if "n_layer" in train_configs:
            decoder_config.n_layer = train_configs["n_layer"]
            decoder.transformer.h = decoder.transformer.h[:decoder_config.n_layer] #! take the bottom layes 
        decoder.transformer.wpe.weight.requires_grad = False
    else:
        # todo: hardcoded for evaluation, remove
        decoder_config.n_layer = 4
        decoder = GPT2PSALMHeadModel(decoder_config)
        decoder.load_state_dict(checkpoints["decoder"])


    if logger is not None:
        logger.info("models created.")
    decoder.to(util.DEVICE).train()
    encoder.to(util.DEVICE).train()

    if (checkpoints is not None) and (not get_only_models):
        return encoder, decoder, checkpoints["optimizer"], checkpoints["step"]
    else:
        return encoder, decoder

def get_optimizer(encoder, decoder, lr: float):
    no_decay = ["bias", "LayerNorm.weight"]     # wd 1e-2 for GPT paper
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in decoder.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': float(lr),
            "weight_decay": 1e-5,
        },
        {
            "params": [p for n, p in decoder.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': float(lr)
        },
        {
            "params": [p for n, p in encoder.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': float(lr),
            "weight_decay": 1e-5,
        },
        {
            "params": [p for n, p in encoder.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': float(lr)
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-8)
    return optimizer

def update_config(train_configs):
    if "val_sample_size" not in train_configs:
        train_configs["val_sample_size"] = 5

    # set default dropout configration if empty.
    if "loss_configs" in train_configs:
        if "main_dropout" not in train_configs["loss_configs"]:
            train_configs["loss_configs"]["main_dropout"] = True
        if "aux_dropout" not in train_configs["loss_configs"]:
            train_configs["loss_configs"]["aux_dropout"] = True
        if "kl_dropouts" not in train_configs["loss_configs"]:
            train_configs["loss_configs"]["kl_dropouts"] = {
                "init_encoder_dropout": True,
                "init_decoder_dropout": True,
                "current_encoder_dropout": True,
                "current_decoder_dropout": True,
            }
        if "reward_weighting" not in train_configs["loss_configs"]:
            train_configs["loss_configs"]["reward_weighting"] = {}
            train_configs["loss_configs"]["reward_weighting"]["method"] = ""
            train_configs["loss_configs"]["reward_weighting"]["weight_type"] = ""

    if "aux_configs" in train_configs:
        if "cap_vals" not in train_configs["aux_configs"]:
            train_configs["aux_configs"]["cap_vals"] = None

    if "freeze_lmhead" not in train_configs["loss_configs"]:
        train_configs["loss_configs"]["freeze_lmhead"] = False

def train(training_arguments: TrainingArgs):
    # load configs
    train_configs = training_arguments.get_config()
    update_config(train_configs)

    # set up a logger
    logger = get_logger(training_arguments)

    # set up wandb
    get_wandb(training_arguments)

    # load tokenizer
    tokenizer = get_tokenizer()

    # load models
    models = get_model(train_configs["pretrained_checkpoint_path"], training_arguments, len(tokenizer.decoder), logger, get_only_models=False)
    if len(models) == 2:
        encoder, decoder = models
        # load optimizer
        optimizer = get_optimizer(encoder, decoder, train_configs["lr"])
        steps = 0
    else:
        encoder, decoder, optimizer_dict, steps = models
        optimizer = get_optimizer(encoder, decoder, train_configs["lr"])
        optimizer.load_state_dict(optimizer_dict)

    # get data loaders
    data_loaders = get_dataset_loaders(logger, training_arguments, train_configs, tokenizer)

    # debug
    if training_arguments.get_no_validation():
        data_loaders["full_val_loader"] = None
        data_loaders["cleaned_val_loader"] = None

    # validate and save model
    if train_configs["skip_validation_at_step_zero"]:
        # skip validation at the begining of epoch 0
        validate_and_save_model(encoder, decoder, optimizer, None, None, tokenizer, 0, training_arguments, logger)
    else:
        validate_and_save_model(encoder, decoder, optimizer, data_loaders["full_val_loader"], data_loaders["cleaned_val_loader"], tokenizer, 0, training_arguments, logger)

    # training loop
    train_loop(encoder, decoder, optimizer, tokenizer, data_loaders, training_arguments, logger, steps)

if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser(description='Train or evaluate a Cereal Bar agent.')
    training_arguments: TrainingArgs = TrainingArgs(parser)
    training_arguments.interpret_args(parser.parse_args())
    train(training_arguments)
