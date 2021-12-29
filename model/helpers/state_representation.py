from __future__ import annotations
from typing import TYPE_CHECKING

import os
import logging
import copy
import numpy as np
import torch
from IPython import embed

from agent.environment import agent
from agent.environment import agent_actions
from agent.environment import card
from agent.environment import environment_objects
from agent.environment import hut
from agent.environment import plant
from agent.environment import position
from agent.environment import rotation
from agent.environment import structures
from agent.environment import terrain
from agent.environment import tree
from agent.environment import util as environment_util

if TYPE_CHECKING:
    from agent.config import state_representation_args
    from agent.data import instruction_example
    from agent.data import partial_observation
    from agent.environment import state_delta
    from typing import List, Tuple, Union

EMPTY_STR: str = '_NONE'

# TODO: name duplicate in modules
class StateRepresentation:
    def __init__(self, breakpoint_type=""):
        """
        State repesentation S + follower's path plan in CerealBar
        Designed to get no more information than agent/simulation/instruction/logical_instruction.py
        """
        # Numpy state representation
        self._static_num_channel = 9
        self._dynamic_num_channel = 6
        self._breakpoint_type = breakpoint_type
        if self._breakpoint_type == "":
            self._planner_channel = 4
        else:
            self._planner_channel = 5

        self._state_num_channel = self._dynamic_num_channel + \
            self._static_num_channel + self._planner_channel
        self._state = np.zeros((environment_util.ENVIRONMENT_WIDTH,
                                environment_util.ENVIRONMENT_DEPTH, self._state_num_channel))

        # Static environment_objects
        self._terrain_indices: List[terrain.Terrain] = [
            EMPTY_STR] + sorted([ter for ter in terrain.Terrain])
        self._str_terrain_indices: List[terrain.Terrain] = [
            EMPTY_STR] + sorted([str(ter) for ter in terrain.Terrain])
        self._prop_type_indices: List[Union[str, environment_objects.ObjectType]] = [EMPTY_STR] + [
            environment_objects.ObjectType.TREE,
            environment_objects.ObjectType.HUT,
            environment_objects.ObjectType.PLANT,
            environment_objects.ObjectType.WINDMILL,
            environment_objects.ObjectType.TOWER,
            environment_objects.ObjectType.TENT,
            environment_objects.ObjectType.LAMPPOST, ]
        self._tree_type_indices: List[Union[str, tree.TreeType]] = [
            EMPTY_STR] + sorted([obj for obj in tree.TreeType])
        self._hut_color_indices: List[Union[str, hut.HutColor]] = [
            EMPTY_STR] + sorted([color for color in hut.HutColor])
        self._hut_rotation_indices: List[Union[str, rotation.Rotation]] = [
            EMPTY_STR] + sorted([rot for rot in rotation.Rotation])
        self._plant_type_indices: List[Union[str, plant.PlantType]] = [
            EMPTY_STR] + sorted([obj for obj in plant.PlantType])
        self._windmill_rotation_indices: List[Union[str, rotation.Rotation]] = [
            EMPTY_STR] + sorted([rot for rot in rotation.Rotation])
        self._tower_rotation_indices: List[Union[str, rotation.Rotation]] = [
            EMPTY_STR] + sorted([rot for rot in rotation.Rotation])
        self._tent_rotation_indices: List[Union[str, rotation.Rotation]] = [
            EMPTY_STR] + sorted([rot for rot in rotation.Rotation])

        # Card (dynamic objects)
        self._card_color_indices: List[Union[card.CardColor, str]] = [
            EMPTY_STR] + sorted([color for color in card.CardColor])
        self._card_shape_indices: List[Union[card.CardShape, str]] = [
            EMPTY_STR] + sorted([shape for shape in card.CardShape])
        self._card_count_indices: List[Union[card.CardCount, str]] = [
            EMPTY_STR] + sorted([count for count in card.CardCount])
        self._card_selection_indices: List[Union[card.CardSelection, str]] = [
            EMPTY_STR] + sorted([selection for selection in card.CardSelection])

        # Leader
        self._leader_rotation_indices: List[Union[rotation.Rotation, str]] = [
            EMPTY_STR] + sorted([rot for rot in rotation.Rotation])

        # Follower's states and plans
        self._follower_rotation_indices: List[Union[rotation.Rotation, str]] = [
            EMPTY_STR] + sorted([rot for rot in rotation.Rotation])
        # follower path one_hot
        # follower goal one_hot

        # Others_tent_rotation_indices
        self._follower_configurations: List[agent.Agent] = []
        self._follower_actions: List[agent_actions.AgentAction] = []

        self.dim_orders = [self._terrain_indices, self._prop_type_indices, self._tree_type_indices, self._hut_color_indices, self._hut_rotation_indices, self._plant_type_indices,
                           self._windmill_rotation_indices, self._tower_rotation_indices, self._tent_rotation_indices, self._card_color_indices, self._card_shape_indices,
                           self._card_count_indices, self._card_selection_indices, self._leader_rotation_indices, self._follower_rotation_indices]

    def get_terrains(self):
        return self._terrain_indices

    def get_prop_types(self):
        return self._prop_type_indices

    def get_tree_types(self):
        return self._tree_type_indices

    def get_hut_color(self):
        return self._hut_color_indices

    def get_hut_rotation(self):
        return self._hut_rotation_indices

    def get_plant_types(self):
        return self._plant_type_indices

    def get_windmill_rotation(self):
        return self._windmill_rotation_indices

    def get_tower_rotation(self):
        return self._tower_rotation_indices

    def get_tent_rotation(self):
        return self._tent_rotation_indices

    def get_card_color(self):
        return self._card_color_indices

    def get_card_shape(self):
        return self._card_shape_indices

    def get_card_count(self):
        return self._card_count_indices

    def get_card_selection(self):
        return self._card_selection_indices

    def get_leader_rotation(self):
        return self._leader_rotation_indices

    def get_follower_rotation(self):
        return self._follower_rotation_indices

    def get_trajectory(self):
        return [EMPTY_STR] + [1]

    def get_goals(self):
        return [EMPTY_STR] + [1]

    def get_avoidance(self):
        return [EMPTY_STR] + [1]

    def get_obstacles(self):
        return [EMPTY_STR] + [1]

    def get_onehot_lf_breakpoints(self):
        return [EMPTY_STR] + [1]

    def get_pose_lf_breakpoints(self):
        return self._follower_rotation_indices

    def get_state_length(self):
        prefix_length = 0
        prefix_length += len(self.get_terrains())
        prefix_length += len(self.get_prop_types())
        prefix_length += len(self.get_tree_types())
        prefix_length += len(self.get_hut_color())
        prefix_length += len(self.get_hut_rotation())
        prefix_length += len(self.get_plant_types())
        prefix_length += len(self.get_windmill_rotation())
        prefix_length += len(self.get_tower_rotation())
        prefix_length += len(self.get_tent_rotation())
        prefix_length += len(self.get_card_color())
        prefix_length += len(self.get_card_shape())
        prefix_length += len(self.get_card_count())
        prefix_length += len(self.get_card_selection())
        prefix_length += len(self.get_leader_rotation())
        prefix_length += len(self.get_follower_rotation())
        prefix_length += len(self.get_trajectory())
        prefix_length += len(self.get_goals())
        prefix_length += len(self.get_avoidance())
        prefix_length += len(self.get_obstacles())
        if self._breakpoint_type == "onehot":
            prefix_length += len(self.get_onehot_lf_breakpoints())
        return prefix_length

    def get_state_num_channel(self):
        return self._state_num_channel

    def get_state_representation(self):
        """
        Return a current state representation
        """
        return self._state

    def get_state_per_configuration(self):
        """
        Return states at each step of follower's configuration
            1. crop a local map and mask the local map by visibility
            2. rotate map to align the orientation (convolutions are rotation invariant)
        """
        return self._state

    def get_follower_actions(self):
        """
        Get ground-tructh action plans for follower's
        """
        return self._follower_actions

    def update_static_representation(self, hexes: List[Tuple[terrain.Terrain, position.Position]], objects: List[environment_objects.EnvironmentObject]):
        """
        Update static representation of the enviornment
        """
        for hex in hexes:
            try:
                self._state[hex[1].x, hex[1].y, 0] = self._terrain_indices.index(hex[0])
            except:
                self._state[hex[1].x, hex[1].y, 0] = self._str_terrain_indices.index(str(hex[0]))

            # self._state [hex[1].x, hex[1].y, 0] = self._state [hex[1].x, hex[1].y, 0] / (len(self._terrain_indices) -1)

        for obj in objects:
            loc = obj.get_position()
            self._state[loc.x, loc.y, 1] = self._prop_type_indices.index(obj.get_type())
            # self._state[ loc.x, loc.y, 1] = self._state[ loc.x, loc.y, 1] / (len(self._prop_type_indices) -1)

            if obj.get_type() == environment_objects.ObjectType.TREE:
                self._state[loc.x, loc.y, 2] = self._tree_type_indices.index(obj.get_tree_type())
                # self._state[ loc.x, loc.y, 2] = self._state[ loc.x, loc.y, 2] / (len(self._tree_type_indices) -1)

            elif obj.get_type() == environment_objects.ObjectType.HUT:
                self._state[loc.x, loc.y, 3] = self._hut_color_indices.index(obj.get_color())
                self._state[loc.x, loc.y, 4] = self._hut_rotation_indices.index(obj.get_rotation())
                # self._state[ loc.x, loc.y, 3] = self._state[ loc.x, loc.y, 3] / (len(self._hut_color_indices) -1)
                # self._state[ loc.x, loc.y, 4] = self._state[ loc.x, loc.y, 4] / (len(self._hut_rotation_indices) -1)

            elif obj.get_type() == environment_objects.ObjectType.PLANT:
                self._state[loc.x, loc.y, 5] = self._plant_type_indices.index(obj.get_plant_type())
                # self._state[ loc.x, loc.y, 5] = self._state[ loc.x, loc.y, 5] / (len(self._plant_type_indices) -1)

            elif obj.get_type() == environment_objects.ObjectType.WINDMILL:
                self._state[loc.x, loc.y, 6] = self._windmill_rotation_indices.index(
                    obj.get_rotation())
                # self._state[ loc.x, loc.y, 6] = self._state[ loc.x, loc.y, 6] / (len(self._windmill_rotation_indices) -1)

            elif obj.get_type() == environment_objects.ObjectType.TOWER:
                self._state[loc.x, loc.y, 7] = self._tower_rotation_indices.index(
                    obj.get_rotation())
                # self._state[ loc.x, loc.y, 7] = self._state[ loc.x, loc.y, 7] / (len(self._tower_rotation_indices) -1)

            elif obj.get_type() == environment_objects.ObjectType.TENT:
                self._state[loc.x, loc.y, 8] = self._tent_rotation_indices.index(obj.get_rotation())
                # self._state[ loc.x, loc.y, 8] = self._state[ loc.x, loc.y, 8]  / (len(self._tent_rotation_indices) -1)


    def _reset_dynamic_representation(self):
        self._state[:, :, self._static_num_channel:] = 0

    def update_dynamic_representation(self, cards: List[card.Card], leader_configuration: agent.Agent, follower_configurations: List[agent.Agent],
                                      best_follower_order: List[card.Card], obstacle_positions: List[position.Position], lfgen_configurations: List[agent.Agent] = None):
        """
        TODO: lfgen_configurations is depreciated
        """
        self._reset_dynamic_representation()
        for card in cards:
            loc = card.get_position()
            self._state[loc.x, loc.y, self._static_num_channel +
                        0] = self._card_color_indices.index(card.get_color())
            self._state[loc.x, loc.y, self._static_num_channel +
                        1] = self._card_shape_indices.index(card.get_shape())
            self._state[loc.x, loc.y, self._static_num_channel +
                        2] = self._card_count_indices.index(card.get_count())
            self._state[loc.x, loc.y, self._static_num_channel +
                        3] = self._card_selection_indices.index(card.get_selection())

        loc = leader_configuration.get_position()
        rot = leader_configuration.get_rotation()
        try:
            self._state[loc.x, loc.y, self._static_num_channel +
                        4] = self._leader_rotation_indices.index(rot)
        except:
             logging.error("index out of range x:{}, y:{}".format(loc.x, loc.y))

        follower = follower_configurations[0]
        loc = follower.get_position()
        rot = follower.get_rotation()
        try:
            self._state[loc.x, loc.y, self._static_num_channel +
                        5] = self._follower_rotation_indices.index(rot)
        except:
             logging.error("index out of range x:{}, y:{}".format(loc.x, loc.y))

        # Follower plan
        for config in follower_configurations:
            loc = config.get_position()
            rot = config.get_rotation()
            self._state[loc.x, loc.y, self._static_num_channel +
                        self._dynamic_num_channel] = 1

        # Goals (target cards)
        for goal in best_follower_order:
            loc = goal.get_position()
            self._state[loc.x, loc.y, self._static_num_channel+self._dynamic_num_channel + 1] = 1

        # Card not to touch (maybe really not necessary)
        tp_cardset = copy.deepcopy(cards)
        for goal in best_follower_order:
            tp_cardset.remove(goal)

        for not_touch in tp_cardset:
            loc = not_touch.get_position()
            self._state[loc.x, loc.y, self._static_num_channel+self._dynamic_num_channel + 2] = 1

        # Obstacles (maybe really not necessary)
        for loc in obstacle_positions:
            self._state[loc.x, loc.y, self._static_num_channel+self._dynamic_num_channel + 3] = 1

        self._follower_configurations = follower_configurations


    def update_other_information(self, follower_actions: List[agent_actions.AgentAction]):
        self._follower_actions = follower_actions
