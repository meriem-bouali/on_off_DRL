import traci
from rl_env.custom_env.env_config import EnvConfig as EC
import numpy as np
import math
from colorama import Fore


from sumo_sim.file_path import FilePath as FP
import os
from sumo_sim.sim_config import SimulationConfig as SC
from rl_env.custom_env.env_config import EnvConfig as EC
import random


class UtilsMixin:
    def set_start_command(self):
        if self.mode == "train":
            self.seed = random.choice(SC.seed_train)

        if self.gui:
            command = ["sumo-gui"]
        else:
            command = ["sumo"]

        command += [
            "-n",
            FP.net_pth,
            "-r",
            FP.rout_pth,
            "--collision.mingap-factor",
            "0",  # only pysical collision will be detected
            "--collision.action",
            "warn",  #  determines what action to take when a collision is detected, default: teleport
            "--xml-validation",
            "never",  # desable XML validation
            "--lanechange.duration",
            "0",
            "--seed",
            self.seed,
        ]

        if self.gui:
            command += [
                "-g",
                FP.view_path,
                "--delay",
                str(SC.delay) if self.mode == "observe" else str(0),  # Use FLOAT in ms as delay between simulation steps
                "--start",
                "true",  # starts the simulation upon opening the gui
                "--quit-on-end",
                "true",  # Quits the GUI when the simulation stops
                "--window-size",
                "1920,1080",  # Maximizes SUMO-GUI window for consistent screenshots
            ]
        if not self.gui:
            command += ["--no-warnings", "true"]  # do not show warnings on terminal

        command += ["--log", "log_message", "--no-step-log", "true"]
        return command

    # get curent and target lanes index
    def get_lane_index(self, veh_id, action):
        return traci.vehicle.getLaneIndex(veh_id), traci.vehicle.getLaneIndex(veh_id) + action - 1

    def has_left_right_lane(self, veh_id):
        lane_id = traci.vehicle.getLaneID(veh_id)
        edge_id = traci.lane.getEdgeID(lane_id)
        # if edge_id[0] == ":":
        #     print("********************************", edge_id)
        lane_index = traci.vehicle.getLaneIndex(veh_id)

        has_left_lane = 1 if lane_index < (traci.edge.getLaneNumber(edge_id) - 1) else 0
        has_right_lane = 1 if lane_index > 0 else 0

        return has_right_lane, has_left_lane

    def get_nb_lanes_position(self):
        """
        Get the number of lanes on the edge where the ego vehicle is currently driving.

        An internal edge (edge IDs starting with ':') lies within an intersection and connects an incoming normal edge with an outgoing normal edge.

        If the vehicle is on an internal edge, return `self.nb_lanes`, which is the number of lanes of the basic road segment (not a merge,
        diverge, or weaving segment), because internal edges do not always reflect the actual number of lanes of the main road.

        If the vehicle is on a normal edge, return the actual number of lanes using `traci.edge.getLaneNumber(edge_id)`.
        """

        edge_id = traci.vehicle.getRoadID(self.ego_id)
        return SC.nb_lanes if edge_id[0] == ":" else traci.edge.getLaneNumber(edge_id)
