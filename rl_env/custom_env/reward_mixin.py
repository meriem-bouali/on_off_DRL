import traci
from rl_env.custom_env.env_config import EnvConfig as EC
from colorama import Fore


class RewardMixin:
    def reward_collision(self, veh_id, is_valide_lc):
        collisions = traci.simulation.getCollidingVehiclesIDList()

        if not is_valide_lc:  # check if vehicle collision
            # print("invalide LC")
            return -30
        elif collisions:  # check if vehicle collision
            self.nb_vehicle_collision += collisions.count(veh_id)
            return collisions.count(veh_id) * -50  # if collisions.count(veh_id)==0, it will return 0
        else:
            return 0
    # def emergency_braking_reward(self, veh_id):
    #     follower = traci.vehicle.getFollower(veh_id)[0]
    #     if traci.vehicle.getAcceleration(veh_id) <= (traci.vehicle.getEmergencyDecel(veh_id) * -1):
    #         self.nb_emrgency_braking+=1
    #         return -8  # emergency braking by AV
    #     # elif follower=='':
    #     #     return "no emergency braking",0
    #     elif follower != "" and traci.vehicle.getAcceleration(follower) <= (traci.vehicle.getEmergencyDecel(follower) * -1):
    #         self.nb_emrgency_braking+=1
    #         return -8  # emergency braking by follower vehicle
    #     else:
    #         return 0  # no emergency braking
        
    def reward_min_gap(self, veh_id):
        leader = traci.vehicle.getLeader(veh_id)
        follower = traci.vehicle.getFollower(veh_id)
        if (leader is None or leader[1] >= 0) and (follower == ("", -1) or follower[1] >= 0):
            return 0
        else:
            self.nb_mingap_violation += 1
            return -15

    # def quality_lc_reward(self, action, distance_current, distance_target):
    #     if action == 1:
    #         return 0
    #     else:
    #         if (
    #             distance_current >= EC.perception_range_front or distance_current <= distance_target <= distance_current + 1
    #         ):  # no gain or only littel gain of 1m
    #             return -2  # useless
    #         elif distance_current > distance_target:
    #             return -4  # harmful
    #         else:
    #             return 1  # useful

    # # # def ping_pong_lc_reward(self, action):
    # # #     if self.prev_action is not None and (self.prev_action + 2 == action or self.prev_action - 2 == action):
    # # #         return -4
    # # #     return 0

    # def is_valide_lc(self, target_lane):
    #     if target_lane >= 0 and target_lane < self.get_nb_lanes_position():
    #         return True
    #     return False

    # get curent and target lanes index
    def get_lane_index(self, veh_id, action):
        return traci.vehicle.getLaneIndex(veh_id), traci.vehicle.getLaneIndex(veh_id) + action - 1

    # def get_LC_gain(self, veh_id, current_lane, target_lane):

    #     mingap = traci.vehicle.getMinGap(veh_id)

    #     leader_curent = traci.vehicle.getLeader(veh_id)
    #     if leader_curent is None:
    #         distance_current = EC.perception_range_front
    #     else:
    #         distance_current = leader_curent[1] + mingap

    #     if target_lane < current_lane:
    #         leader_target = traci.vehicle.getRightLeaders(veh_id)
    #     elif target_lane > current_lane:
    #         leader_target = traci.vehicle.getLeftLeaders(veh_id)
    #     else:
    #         raise ValueError("is lane keeping not LC")  # target_lane==current_lane

    #     if leader_target == ():
    #         distance_target = EC.perception_range_front
    #     else:
    #         distance_target = leader_target[0][1] + mingap

    #     return (distance_current, distance_target)

    def reward_function(self, veh_id, action, is_valide_lc): #, distance_current, distance_target):
        rwd_collision = self.reward_collision(self.ego_id, is_valide_lc)
        if rwd_collision:
            # print("reward= ", rwd_collision)
            return rwd_collision

        else:
            rwd_mingap = self.reward_min_gap(veh_id)
            # rwd_emergency_braking = self.emergency_braking_reward(veh_id)
            rwd_perform_lc = (action == 0 or action == 2) * -1
            rwd_speed = round((self.speed-33.3) / 33.3, 2)  # 33.3 is speed limit
            rwd_destination = (self.terminated()) * 20
            return rwd_mingap + rwd_perform_lc  + rwd_speed + rwd_destination  # # # + rwd_ping_pong
