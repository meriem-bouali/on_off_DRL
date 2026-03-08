# import sys


from rl_env.custom_env.env_config import EnvConfig as EC


import traci
import traci.constants as tc


import numpy as np
from colorama import Fore
import uuid

import random
from sumo_sim.sim_config import SimulationConfig as SC

from rl_env.custom_env.reward_mixin import RewardMixin
from rl_env.custom_env.utils_mixin import UtilsMixin
from dqn.dqn_config import HYPER_PARAMS


from sumo_sim.file_path import FilePath as FP
import os
# from rl_env.custom_env.state_speedmatrix_mixin import StateMixin


class SumoEnv(RewardMixin, UtilsMixin):
    def __init__(self, mode, policy, gui=False):
        self.policy = policy
        self.gui = gui
        self.ego_id = SC.ego_veh_id
        self.episode_length = EC.episode_max_length  # + SC.ego_veh_entry_step
        self.mode = mode
        self.possible_actions = np.array([0, 1, 2])  # right, keep, left
        self.action_space_n = HYPER_PARAMS.action_n
        self.observation_space_n = HYPER_PARAMS.observation_n
        # // self.liste_reward = []

    def get_observation(self):
        has_right_lane, has_left_lane = self.has_left_right_lane(self.ego_id)

        leader = traci.vehicle.getLeader(self.ego_id)
        follower = traci.vehicle.getFollower(self.ego_id)
        left_leader = traci.vehicle.getLeftLeaders(self.ego_id)
        left_follower = traci.vehicle.getLeftFollowers(self.ego_id)
        right_leader = traci.vehicle.getRightLeaders(self.ego_id)
        right_follower = traci.vehicle.getRightFollowers(self.ego_id)

        ## get surrounding vehicles IDs and gap between AV and surrounding vehicles
        leader_id, leader_gap = (leader[0], round((leader[1] + 2.5) / EC.perception_range_front, 2)) if leader != None else (None, -2.0)
        leader_id, leader_gap = (None, -2) if leader_gap > 1 else (leader_id, leader_gap)

        follower_id, follower_gap = (
            (follower[0], round((follower[1] + 2.5) / EC.perception_range_front, 2)) if follower != ("", -1.0) else (None, -2.0)
        )
        follower_id, follower_gap = (None, -2) if follower_gap > 1 else (follower_id, follower_gap)

        left_leader_id, left_leader_gap = (
            (left_leader[0][0], round((left_leader[0][1] + 2.5) / EC.perception_range_front, 2)) if left_leader != () else (None, -2.0)
        )
        left_leader_id, left_leader_gap = (None, -2) if left_leader_gap > 1 else (left_leader_id, left_leader_gap)

        left_follower_id, left_follower_gap = (
            (left_follower[0][0], round((left_follower[0][1] + 2.5) / EC.perception_range_front, 2))
            if left_follower != ()
            else (None, -2.0)
        )
        left_follower_id, left_follower_gap = (None, -2) if left_follower_gap > 1 else (left_follower_id, left_follower_gap)

        right_leader_id, right_leader_gap = (
            (right_leader[0][0], round((right_leader[0][1] + 2.5) / EC.perception_range_front, 2)) if right_leader != () else (None, -2.0)
        )
        right_leader_id, right_leader_gap = (None, -2) if right_leader_gap > 1 else (right_leader_id, right_leader_gap)

        right_follower_id, right_follower_gap = (
            (right_follower[0][0], round((right_follower[0][1] + 2.5) / EC.perception_range_front, 2))
            if right_follower != ()
            else (None, -2.0)
        )
        right_follower_id, right_follower_gap = (None, -2) if right_follower_gap > 1 else (right_follower_id, right_follower_gap)

        # # extract relative speeds
        ego_veh_speed = traci.vehicle.getSpeed(self.ego_id)
        leader_relatif_s = round((ego_veh_speed - traci.vehicle.getSpeed(leader_id)) / 33.33, 2) if leader_id != None else -2
        follower_relatif_s = round((ego_veh_speed - traci.vehicle.getSpeed(follower_id)) / 33.33, 2) if follower_id != None else -2
        left_leader_relatif_s = round((ego_veh_speed - traci.vehicle.getSpeed(left_leader_id)) / 33.33, 2) if left_leader_id != None else -2
        left_follower_relatif_s = (
            round((ego_veh_speed - traci.vehicle.getSpeed(left_follower_id)) / 33.33, 2) if left_follower_id != None else -2
        )
        right_leader_relatif_s = (
            round((ego_veh_speed - traci.vehicle.getSpeed(right_leader_id)) / 33.33, 2) if right_leader_id != None else -2
        )
        right_follower_relatif_s = (
            round((ego_veh_speed - traci.vehicle.getSpeed(right_follower_id)) / 33.33, 2) if right_follower_id != None else -2
        )

        ego_position = traci.vehicle.getPosition(self.ego_id)[0]
        driving_in_weaving = (
            SC.on_ramp_position[0] <= ego_position <= SC.off_ramp_position[0]
            or SC.on_ramp_position[1] <= ego_position <= SC.off_ramp_position[1]
        ) * 1

        valid_distances_onramp = [
            ramp_x - ego_position for ramp_x in SC.on_ramp_position if -50 <= (ramp_x - ego_position) <= EC.perception_range_front
        ]

        valid_distances_offramp = [
            ramp_x - ego_position for ramp_x in SC.off_ramp_position if -50 <= (ramp_x - ego_position) <= EC.perception_range_front
        ]

        # Return nearest in absolute value or -2 if none
        dist_to_onramp = round(min(valid_distances_onramp, key=abs) / EC.perception_range_front, 2) if valid_distances_onramp else -2
        dist_to_offramp = round(min(valid_distances_offramp, key=abs) / EC.perception_range_front, 2) if valid_distances_offramp else -2

        return [
            has_right_lane,
            has_left_lane,
            driving_in_weaving,
            dist_to_onramp,
            dist_to_offramp,
            leader_gap,
            leader_relatif_s,
            follower_gap,
            follower_relatif_s,
            left_leader_gap,
            left_leader_relatif_s,
            left_follower_gap,
            left_follower_relatif_s,
            right_leader_gap,
            right_leader_relatif_s,
            right_follower_gap,
            right_follower_relatif_s,
        ]

    def start(self):
        if not traci.isLoaded():
            start_cmd = self.set_start_command()

            # start simulation
            traci.start(start_cmd)

            # Run 10 simulation steps
            traci.simulationStep(SC.ego_veh_entry_step)

            # add ego vehicle into simulation
            traci.vehicle.add(
                self.ego_id,
                routeID=SC.ego_veh_route_id,
                typeID=SC.ego_veh_vtype_id,
                depart=SC.ego_veh_entry_step,
                departLane="random",
                arrivalLane="random",
            )

            # Check if the vehicle is added to the simulation. When traffic flow is high,
            # it may not be possible to add the vehicle at the requested time step,
            # but it might be added in future time steps.
            traci.simulationStep(SC.ego_veh_entry_step + 1)
            while self.ego_id not in traci.vehicle.getIDList():
                traci.simulationStep()
            # To avoid having a 50m empty gap behind the AV —which could disturb training—
            # step the simulation until the AV reaches the starting position which corresponds to 50+5m from the beginning of the road.
            while not traci.vehicle.getPosition(self.ego_id)[0] >= 55:
                traci.simulationStep()
            self.speed = traci.vehicle.getSpeed(self.ego_id)

    def stop(self):
        if traci.isLoaded():
            traci.close()

    def reset(self):
        # reset traking var
        self.time_step = 0
        self.total_reward = 0
        self.nb_lc = 0
        self.nb_mingap_violation = 0
        self.nb_emrgency_braking = 0
        self.nb_invalide_lc = 0
        self.nb_vehicle_collision = 0  # nb of vehicle in collision with AV not nb of collision
        self.ping_pong = 0
        self.total_speed = 0
        # // self.liste_reward = []

        self.speed = None
        self.prev_speed = None
        self.prev_action = None
        # self.seed = None

        self.stop()
        self.start()
        self.obs = self.get_observation()
        return self.obs

    def step(self, action):

        current_lane, target_lane = self.get_lane_index(self.ego_id, action)
        # print(Fore.YELLOW,"current_lane=",current_lane, "target_lane=",target_lane, Fore.RESET)

        self.time_step += 1

        has_right_lane, has_left_lane = self.obs[0:2]  # self.has_left_right_lane(self.ego_id)  #

        is_valide_lane = (action == 0 and has_right_lane) or (action == 2 and has_left_lane) or (action==1)

        # if is_valide_lane and action != 1:
        #     (distance_current, distance_target) = self.get_LC_gain(self.ego_id, current_lane, target_lane)
        # else:
        #     (distance_current, distance_target) = None, None
        self.apply_action(self.ego_id, target_lane, is_valide_lane, action)
        self.obs = self.get_observation()
        self.speed = traci.vehicle.getSpeed(self.ego_id)
        reward = self.reward_function(self.ego_id, action, is_valide_lane) #, distance_current, distance_target)

        self.total_reward += reward

        self.update_tracking_metrics(action, is_valide_lane)

        done = self.truncated() or self.terminated()
        info = {"l": self.time_step, "r": self.total_reward}
        if done:
            self.set_episode_metrics(info)

        self.prev_action = action  # for ping pong action check
        return (
            self.obs,
            reward,
            done,
            info,
        )

    def terminated(self) -> bool:
        return traci.vehicle.getPosition(self.ego_id)[0] >= 5000  #  An episode terminate if the ego vehicle reach the distination

    def truncated(self) -> bool:
        return self.time_step >= self.episode_length

    def apply_action(self, veh_id, target_lane, is_valide_lane, action):
        traci.vehicle.setLaneChangeMode(veh_id, 0)
        if is_valide_lane and (action == 0 or action == 2):
            traci.vehicle.changeLane(veh_id, target_lane, duration=0)
        traci.simulationStep()

    def update_tracking_metrics(self, action, is_valide_lane):
        if action == 0 or action == 2:
            self.nb_lc += 1
            if self.prev_action is not None and (self.prev_action + 2 == action or self.prev_action - 2 == action):
                self.ping_pong += 1
        self.nb_invalide_lc += not is_valide_lane
        self.total_speed += traci.vehicle.getSpeed(self.ego_id)
        ## Other metrics were updated during reward function computation.

    def set_episode_metrics(self, info):
        info["episode_length"] = self.time_step
        info["episode_total_reward"] = self.total_reward
        info["episode_nb_lc"] = self.nb_lc
        info["episode_nb_mingap_violation"] = self.nb_mingap_violation
        info["episode_nb_emrgency_braking"] = self.nb_emrgency_braking
        info["episode_nb_invalide_lc"] = self.nb_invalide_lc
        info["episode_nb_vehicle_collision"] = self.nb_vehicle_collision
        info["episode_nb_ping_pong"] = self.ping_pong
        info["episode_avg_speed"] = round(self.total_speed / self.time_step, 2)
        info["seed"] = self.seed

        ### for epiosde count will be added by Agent class
        return info


# print(__file__)
