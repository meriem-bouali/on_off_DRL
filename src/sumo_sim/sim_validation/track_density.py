import sumolib
from sumo_sim.file_path import FilePath as FP
from sumo_sim.sim_config import SimulationConfig as SC
from rl_env.custom_env.env_config import EnvConfig as EC
from datetime import datetime
import traci
import os
import csv


class TrackDensity:
    def __init__(self):
        self.main_edges_set = ["0to1", "1to2", "2to3", "3to4", "4to5"]
        self.freq = 10
        self.nb_lanes = 3
        self.edges_length, self.edges_nb_lanes = self.get_main_edge_length()
        # self.dt = datetime.now().strftime("%d-%m-%Y")  # "%d-%m-%Y_(%H-%M)"

    def get_main_edge_length(self):
        """
        Retrieve the length of each main-stream edge in the SUMO network.

        Returns:
            dict: A dictionary where keys are edge IDs (str) and values are
                edge lengths (float).
        """
        # Load the network
        net = sumolib.net.readNet(FP.net_pth)

        # Get all edges
        edges = net.getEdges()
        edges_length = {}
        edges_nb_lanes = {}

        for edge in edges:
            if edge.getID() in self.main_edges_set:
                edges_length[edge.getID()] = edge.getLength()
                edges_nb_lanes[edge.getID()] = len(edge.getLanes())

        return edges_length, edges_nb_lanes

    def PCperM_to_PCperMi(self, density_pc_m):
        return density_pc_m * 1609.34

    def get_edge_density(self,seed,tag):  # get density in pc/mi/ln

        path=os.path.join(FP.sim_validator_pth,"csv", "density_"+tag)
        os.makedirs(path, exist_ok=True)
        
        traci.start(
            [
                "sumo",
                "-n",
                FP.net_pth,
                "-r",
                FP.rout_pth,
                # "--gui-settings-file", FP.view_path,
                # "--delay", "10",  # Adds [FLOAT] delay between simulation steps
                # '--lanechange.duration', "1",  # to set the LC duration to 2 seconds
                "--start",
                "true",
                "--quit-on-end",  # Quits the GUI when the simulation stops
                "true",
                "--xml-validation",
                "never",  # desable XML validation
                # "--log",
                # "log",
                "--seed",
                seed,
            ]
        )

        traci.simulationStep(SC.ego_veh_entry_step)
        file_pth = os.path.join(path, f"density_{seed}.csv")
        with open(file_pth, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Timestep"] + self.main_edges_set)
            i = SC.ego_veh_entry_step
            while i <= (SC.ego_veh_entry_step + EC.episode_max_length):
                traci.simulationStep()
                if i % self.freq == 0:
                    row = [i]  # first column = timestep

                    for edge_id in self.main_edges_set:
                        # density pc/m
                        density_m = traci.edge.getLastStepVehicleNumber(edge_id) / self.edges_length[edge_id]

                        # pc/mi
                        density_mi = self.PCperM_to_PCperMi(density_m)

                        # pc/mi/ln
                        density_mi_ln = density_mi / self.edges_nb_lanes[edge_id]

                        row.append(density_mi_ln)

                    writer.writerow(row)

                i += 1
        traci.close()


if __name__ == "__main__":
    track_density = TrackDensity()
    for seed in SC.seed_train:
        track_density.get_edge_density(seed=seed,tag="seed_train")
    for seed in SC.seed_test:
        track_density.get_edge_density(seed=seed,tag="seed_test")
