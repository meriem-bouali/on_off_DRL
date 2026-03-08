from dataclasses import dataclass
import os


@dataclass
class FilePath:
    # ____________________ SUMO Simulation  data ____________________
    sim_data_pth = os.path.join(os.path.dirname(__file__), "data")

    net_pth = os.path.join(sim_data_pth, "network.net.xml")  # Network files path
    edg_pth = os.path.join(sim_data_pth, "edge.edg.xml")  # Edge files  path
    edg_type_pth = os.path.join(sim_data_pth, "edge_type.type.xml")  # Edge_type files path
    nod_pth = os.path.join(sim_data_pth, "nodes.nod.xml")  # Nodes XML file path

    rout_pth = os.path.join(sim_data_pth, "demand.rou.xml")  # Routes files path

    view_path = os.path.join(sim_data_pth, "view_setting.xml")

    # # ____________________data collector ____________________
    # data_collector_pth = os.path.join(sim_data_pth, "data_collector.add.xml")
    # detector_pth = os.path.join(sim_data_pth, "detector.add.xml")

    # ____________________Simulation validation data ____________________
    sim_validator_pth = os.path.join(os.path.dirname(__file__), "sim_validation")  # Validation data directory
    # sim_validator_lanedata_pth = os.path.join(sim_validator_pth, "traffic_measures", "lanedata", "")  # lanedata files
    # sim_validator_edgedata_pth = os.path.join(sim_validator_pth, "traffic_measures", "edgedata", "")  # edgedata files
    # sim_validator_density_pth = os.path.join(sim_validator_pth, "traffic_measures", "Track_density", "")  # density csv and plot files
    # sim_validator_detector_pth = os.path.join(sim_validator_pth, "traffic_measures", "detector", "")  # detector output files

    # sim_validator_screenshot_pth = os.path.join(sim_validator_pth, "screenshot", "")  #


if __name__ == "__main__":
    print(FilePath.sim_validator_pth)
