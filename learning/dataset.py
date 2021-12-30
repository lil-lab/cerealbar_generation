import os, sys
import pickle
import random
import numpy as np
from typing import Dict, Any, List, Set, Tuple
from IPython import embed

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

from model import util
from model import hex_util
from model import state_representation
from agent.environment import agent, position, rotation
from learning.utils import print_and_log


class INSTRUCTION_DATASET(Dataset):
    def  __init__(self, tokenizer: GPT2Tokenizer, logger):
        self._states: List[torch.tensor] = []
        self._trajs: List[torch.tensor] = []
        self._instrs: List[torch.tensor] = []
        self._data_name: List[int] = []
        self._data_label: List[int] = []
        self._rewards: List[torch.tensor] = []
        self._tokenizer = tokenizer
        self._logger = logger

        offset_index_tensor = hex_util._get_batched_offset_index_tensor(
            1, 25, 25)
        additional_size = (25 - 1) // 2
        self._axial_index_tensor = hex_util._get_axial_index_tensor(
            offset_index_tensor, add_u=additional_size)

    def _get_axial_traj_from_suhr_etal(self, agent_states: List[List]) -> torch.tensor:
        traj = []
        for ag in agent_states:
            pos = ag.get_position()
            rot = ag.get_rotation()
            rot = self._state_rep_object._follower_rotation_indices.index(rot)
            row, col = self._axial_index_tensor[0, pos.x, pos.y, :]
            row, col = row.item(), col.item()
            traj.append(np.array([row, col, rot]))
        traj = np.stack(traj)
        traj = torch.tensor(traj)
        return traj

    # TODO: remove code duplication
    def _offset_numpy_to_axial_pytorch(self, state_npy: np.array) -> torch.tensor:
        state_tensor = torch.tensor(state_npy).to(util.DEVICE).float()
        state_tensor = state_tensor.unsqueeze(0)
        state_tensor = state_tensor.permute(0, 3, 1, 2)
        state_tensor = hex_util.offset_tensor_to_axial(state_tensor)
        state_tensor = state_tensor.permute(0, 2, 3, 1)
        if torch.cuda.is_available(): # TODO: come up with a better soluction to put data in cpu memory
            state_tensor = state_tensor.cpu()
        return state_tensor

    def _read_game_ids(self, game_id_files: str = None):
        if game_id_files is None:
            self._valid_game_ids = None
        else:
            with open(game_id_files, "r") as infile:
                self._valid_game_ids = set([l.strip() for l in infile.readlines()])

    def _sample_game_ids(self, data: Dict):
        if self._valid_game_ids is None:
            self._valid_game_ids = set()
            for data_idx in data.keys():
                datum = data[data_idx]
                game_id = datum["example_key"].split("-")[0]
                self._valid_game_ids.add(game_id)

        if self._max_games >= len(self._valid_game_ids):
            self._sampled_game_ids = self._valid_game_ids
        else:
            self._sampled_game_ids = random.sample(self._valid_game_ids, self._max_games)
            print_and_log("{} games found.".format(len(self._valid_game_ids)), self._logger)
            print_and_log("{} games sampled.".format(self._max_games), self._logger)

            for game_id in self._sampled_game_ids:
                print_and_log("    {}".format(game_id), self._logger)

    def _get_axial_traj_from_human_feedback(self, follower_tuples: List[List]) -> torch.tensor:
        traj = []
        for tp in follower_tuples:
            x, y, rot = tp
            row, col = self._axial_index_tensor[0, x, y, :]
            row, col = row.item(), col.item()
            traj.append(np.array([row, col, rot]))
        traj = np.stack(traj)
        traj = torch.tensor(traj)
        return traj

    def _get_user_state_rep(self, state_rep: np.array, trajectory):
        # update follower trajectory
        # WARNING: visitation will be 0/1-ed in encoder, therefore we will discrtize them here.
        state_rep[..., self._state_rep_object._static_num_channel + self._state_rep_object._dynamic_num_channel] = 0
        for tj in trajectory:
            x, y, _ = tj
            state_rep[x, y, self._state_rep_object._static_num_channel + self._state_rep_object._dynamic_num_channel] = 1

        # update card to touch / card not to touch if necessary
        card_locs = np.argwhere(state_rep[:,:,self._state_rep_object._static_num_channel] != 0)
        card_positions: List[position.Position] = [position.Position(loc[0], loc[1]) for loc in card_locs]
        state_rep[..., self._state_rep_object._static_num_channel + self._state_rep_object._dynamic_num_channel + 1] = 0
        state_rep[..., self._state_rep_object._static_num_channel + self._state_rep_object._dynamic_num_channel + 2] = 0

        moved = False
        follower_effective_pos = []
        for tj in trajectory:
            if not np.array_equal(trajectory[0][:1], tj[:1]):
                moved = True
            if moved:
                follower_effective_pos.append(position.Position(tj[0], tj[1]))

        card_to_touch = []
        card_not_to_touch = []
        for pos in card_positions:
            if pos in set(follower_effective_pos):
                # card to touch
                state_rep[pos.x, pos.y, self._state_rep_object._static_num_channel + self._state_rep_object._dynamic_num_channel + 1] = 1
                card_to_touch.append(pos)
            else:
                # card not to touch
                state_rep[pos.x, pos.y, self._state_rep_object._static_num_channel + self._state_rep_object._dynamic_num_channel + 2] = 1
                card_not_to_touch.append(pos)

        return state_rep

    def __getitem__(self, index):
        """
        Generates one sample of data
        """
        # select sample
        state = self._states[index]
        instruction = self._instrs[index]
        trajectory = self._trajs[index]
        data_idx = self._data_name[index]
        reward = self._rewards[index]
        data_label = self._data_label[index]
        outputs = ()
        outputs += (state,)
        outputs += (instruction,)
        outputs += (trajectory,)
        outputs += (data_idx,)
        outputs += (reward,)
        outputs += (data_label,)

        return outputs

    def __len__(self):
        return len(self._data_name)

    def add_data_from_suhr_etal(self, pkl_path, reward_fnc, data_label: str ="", max_examples: int = np.inf, max_games: int = np.inf, game_id_files: str = None):
        self._max_games = max_games
        self._read_game_ids(game_id_files)

        data = pickle.load(open(pkl_path, "rb"))
        self._sample_game_ids(data)
        self._selected_game_ids = set()

        self._state_rep_object = state_representation.StateRepresentation()
        data_keys = list(data.keys())
        if len(data_keys) > max_examples:
            random.shuffle(data_keys) # To sample data from diverese game_ids, since the original data is ordered accoriding to game_id
        num_data_added: int = 0

        for data_idx in data_keys:
            datum = data[data_idx]
            instr = datum["instruction"]
            state = datum["state"]
            trajectory = datum["trajectory"]
            game_id = datum["example_key"].split("-")[0]

            # Limit the number of training examples with two criterion
            #   1. Number of training examples
            #   2. Number of uqnique game ids
            if game_id not in self._sampled_game_ids:
                continue

            instr_idx = [self._tokenizer.bos_token_id] + \
                self._tokenizer.encode(instr) + [self._tokenizer.bos_token_id]
            instr_idx = torch.tensor(instr_idx)
            state = self._offset_numpy_to_axial_pytorch(state)
            traj_tensors = self._get_axial_traj_from_suhr_etal(trajectory)
            if reward_fnc is not None:
                reward = reward_fnc(data)

            self._instrs.append(instr_idx)
            self._states.append(state)
            self._data_name.append("suhr_etal_" + str(data_idx))
            self._trajs.append(traj_tensors)
            self._selected_game_ids.add(game_id)
            self._rewards.append(reward)
            self._data_label.append(data_label)

            num_data_added += 1
            if num_data_added % 100 == 0:
                print("{} instance loaded".format(num_data_added))
            if num_data_added >= max_examples:
                break

        if self._logger is not None:
            print_and_log("total {} examples.".format(len(self._instrs)), self._logger)
            print_and_log("total {} game.".format(len(self._selected_game_ids)), self._logger)

    def add_data_from_human_feedback(self, filename: str, reward_fnc, data_label: str = "", max_examples: int = np.inf):
        num_data_added: int = 0
        infile = open(filename, "r")
        self._state_rep_object = state_representation.StateRepresentation()

        for line in infile.readlines():
            line = line.strip()
            pkl_name: str = line.split()[0]
            traj_type: str = line.split()[1]
            try:
                data = pickle.load(open(pkl_name, "rb"))
            except:
                print_and_log("{} not found.".format(pkl_name), self._logger)
            pkl_name = pkl_name.replace(".pkl", "_{}.pkl".format(traj_type))
            state_rep = data["state_representation"]
            try:
                # calcualte reward before modifying state
                if reward_fnc is not None:
                    reward = reward_fnc(data)

                if traj_type=="ground-truth":
                    trajectory = data["planned_offset_tuples"]
                else:
                    trajectory = data["user_offset_tuples"]
                    state_rep = self._get_user_state_rep(state_rep, trajectory)

                state_tensor = self._offset_numpy_to_axial_pytorch(state_rep)
                traj_tensor = self._get_axial_traj_from_human_feedback(trajectory)
                instruction_tensor = torch.tensor(data["tokenized_instruction"][0])

                self._states.append(state_tensor)
                self._trajs.append(traj_tensor)
                self._instrs.append(instruction_tensor)
                self._data_name.append(pkl_name)
                self._rewards.append(reward)
                self._data_label.append(data_label)
                num_data_added += 1
                if num_data_added % 100 == 0:
                    print("{} instance loaded".format(num_data_added))
                if num_data_added >= max_examples:
                    break
            except:
                print("Something went wrong with {}.".format(pkl_name))


        if len(self._rewards) > 0:
            assert(len(self._states) == len(self._trajs) == len(self._instrs) == len(self._data_name) == len(self._rewards))
            print_and_log("Reward: min {}, max {} mean {} +- std {}".format(torch.min(torch.stack(self._rewards)).item(), torch.max(torch.stack(self._rewards)).item(), torch.mean(torch.stack(self._rewards)).item(), torch.std(torch.stack(self._rewards)).item()), self._logger)


class ADDITIONAL_INSTRUCTION_DATASET(INSTRUCTION_DATASET):
    def  __init__(self, tokenizer: GPT2Tokenizer, logger):
        self._states: List[torch.tensor] = {}
        self._trajs: List[torch.tensor] = {}
        self._instrs: List[torch.tensor] = {}
        self._data_name: List[int] = {}
        self._rewards: List[torch.tensor] = {}
        self._tokenizer = tokenizer
        self._logger = logger

        offset_index_tensor = hex_util._get_batched_offset_index_tensor(
            1, 25, 25)
        additional_size = (25 - 1) // 2
        self._axial_index_tensor = hex_util._get_axial_index_tensor(
            offset_index_tensor, add_u=additional_size)

    def add_data(self, data_path: str, reward_fnc=None):
        infile = open(data_path, "r")
        self._state_rep_object = state_representation.StateRepresentation()

        for line in infile.readlines():
            line = line.strip()
            pkl_name: str = line.split()[0]
            traj_type: str = line.split()[1]
            try:
                data = pickle.load(open(pkl_name, "rb"))
            except:
                print_and_log("{} not found.".format(pkl_name), self._logger)
            state_rep = data["state_representation"]
            try:
                # calcualte reward before modifying state
                if reward_fnc is not None:
                    reward = reward_fnc(data)

                if traj_type=="ground-truth":
                    trajectory = data["planned_offset_tuples"]
                else:
                    trajectory = data["user_offset_tuples"]
                    state_rep = self._get_user_state_rep(state_rep, trajectory)

                state_tensor = self._offset_numpy_to_axial_pytorch(state_rep)
                traj_tensor = self._get_axial_traj_from_human_feedback(trajectory)
                instruction_tensor = torch.tensor(data["tokenized_instruction"][0])

                traj_pkl_name = pkl_name.replace(".pkl", "_{}.pkl".format(traj_type))
                self._states[traj_pkl_name] = state_tensor
                self._trajs[traj_pkl_name] = traj_tensor
                self._instrs[traj_pkl_name] = instruction_tensor
                self._data_name[traj_pkl_name] = traj_pkl_name
                if reward_fnc is not None:
                    self._rewards[traj_pkl_name] = reward
            except:
                print("Something went wrong with {}.".format(traj_pkl_name))


    def add_scores(self, score_path):
        self._query_scores = pickle.load(open(score_path, "rb"))

    def get_values(self, key, batch_size: int, sample_method: str):
        #! n-gram examples have the examples less than batch_size sometimes
        batch_size = min(len(self._query_scores[key].keys()), batch_size)

        # get positive instructions
        if sample_method == "topk":
            pos_file_names = list(self._query_scores[key].keys())[:batch_size]
        elif sample_method == "sample":
            pos_file_names = random.sample(list(self._query_scores[key].keys())[:100], batch_size)
        else:
            raise ValueError("sample_method {} is currently not supported.".format(sample_method))
        query_results = []
        for f in pos_file_names:
            traj_f = f.replace(".pkl", "_{}.pkl".format("ground-truth"))
            if traj_f not in self._data_name.keys():
                traj_f = f.replace(".pkl", "_{}.pkl".format("user"))
            query_results.append(traj_f)
        return query_results


    def collate_results(self, query_results):
        inputs, instructions, trajectories, data_ids, rewards = [], [], [], [], []
        for k in query_results:
            inputs.append(self._states[k])
            instructions.append(self._instrs[k])
            trajectories.append(self._trajs[k])
            data_ids.append(self._data_name[k])
            rewards.append(self._rewards[k])

        input_tensors = torch.cat([input for input in inputs], 0)
        instruction_tensors = [instr for instr in instructions]
        instruction_tensors = pad_sequence(instruction_tensors, 1, padding_value=0) #? padding value is 0
        trajectories_tensors = [traj for traj in trajectories]
        rewards = torch.cat([reward for reward in rewards], 0)
        return input_tensors, instruction_tensors, trajectories_tensors, data_ids, rewards


    def get_batch(self, query_opt, key):
        batch_size, query_method, sample_method = query_opt["batch_size"], query_opt["query_method"], query_opt["sample_method"]

        if query_method == "sentence":
            # sample from possible options
            if sample_method == "random":
                query_results = random.sample(list(self._data_name.keys()), batch_size)
            else:
                assert(len(key) == 1)
                key = "_".join(key[0].split("_")[:-1])
                key += ".pkl"
                query_results = self.get_values(key, batch_size, sample_method)
        else:
            raise ValueError("query_method {} is currently not supported.".format(query_method))

        # return batch from query_results
        return self.collate_results(query_results)


def merge_batch(batch1, batch2):
    input_tensors1, instruction_tensors1, trajectories_tensors1, data_ids1, rewards1 = batch1
    input_tensors2, instruction_tensors2, trajectories_tensors2, data_ids2, rewards2 = batch2
    merged_input_tensors = torch.cat([input_tensors1, input_tensors2], 0)
    merged_instruction_tensors = [instruction_tensors1[0,:]] + [instruction_tensors2[i ,:] for i in range(instruction_tensors2.shape[0])]
    merged_instruction_tensors = pad_sequence(merged_instruction_tensors, 1, padding_value=0) #? padding value is 0
    merged_trajectories_tensors = trajectories_tensors1 + trajectories_tensors2
    merged_data_ids= tuple(list(data_ids1) + list(data_ids2))
    merged_rewards = torch.cat([rewards1, rewards2], 0)

    return merged_input_tensors, merged_instruction_tensors, merged_trajectories_tensors, merged_data_ids, merged_rewards

def collate_fn(data):
    """
    build mini-batch tensors from a list of (image, caption) tuples
    """
    data.sort(key=lambda x: len(x[1]), reverse=True)
    zipped_data = list(zip(*data))
    inputs, instructions, trajectories, data_ids, rewards, data_labels = zipped_data

    # nunmpy to tensor
    input_tensors = torch.cat([input for input in inputs], 0)
    instruction_tensors = [instr for instr in instructions]
    instruction_tensors = pad_sequence(instruction_tensors, 1, padding_value=0) #? padding value is 0
    trajectories_tensors = [traj for traj in trajectories]
    rewards = torch.cat([reward for reward in rewards], 0)
    return input_tensors, instruction_tensors, trajectories_tensors, data_ids, rewards, data_labels
