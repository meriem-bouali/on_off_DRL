from dataclasses import dataclass


@dataclass
class SimulationConfig:
    delay = 150
    colored_route = False
    colored_veh = True
    ego_veh_id = "ego_veh"
    ego_veh_vtype_id = "ego_veh_vtype"
    ego_veh_route_id = "ego_veh_route"
    veh_length = "5.0"
    veh_type = "passenger"
    ego_veh_entry_step = 500  # the step num at which the ego vehicle enters the simulation
    nb_lanes = 3
    on_ramp_position = [965, 2965]  # On-ramp X positions identified after building the SUMO network
    off_ramp_position = [2036, 4033]  # Off-ramp X positions identified after building the SUMO network

    seed_train=["23423","42", "123", "256", "314", "1337", "1500", "1729"]
    seed_test=["11149", "20952", "8024", "25018", "5231", "2848", "24132", "3648", "29234", "7055", "6515", "17856", "7164", "16559", "19309", "7623", "3358", "7223", "13848", "13825", "11029", "21295", "18390", "9115", "976", "1041", "24270", "9105", "23462", "22981", "819", "3070", "869", "19726", "17870", "4572", "24299", "9012", "24864", "13746", "28485", "212", "26523", "14719", "26405", "7314", "5094", "22876", "19349", "22174"]

    @dataclass
    class vTypeColor:
        aggressive = "#00FFFF"
        normal = "#FF4500"
        conservative = "#32CD32"
        ego = "magenta"
