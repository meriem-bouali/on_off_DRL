import os
from csv import DictWriter
import csv
from torch.utils.tensorboard.writer import SummaryWriter
import json

from tensorboard.backend.event_processing import event_accumulator
from datetime import datetime

from dqn import  HYPER_PARAMS

class Logger:
    def __init__(self, algo, mode, policy, store_trs=True, store_episode_metrics=False, extra=None):
        # Create a timestamp string for unique file naming
        time = datetime.now().strftime("(%Y-%m-%d_%H-%M)")

        # TRAINING MODE LOGGING SETUP
        if mode == "train" and store_episode_metrics == True:
            # Path for episode metrics (TensorBoard)
            if policy == "OnRL" or policy == "OnOnRL": # to check 
                self.log_episode_metrics_path = os.path.join(
                    HYPER_PARAMS.log_dir, f"{mode}_episode_metrics", algo + "_" + policy
                )
                self.summary_writer_metrics_eps = SummaryWriter(self.log_episode_metrics_path)

        
        if mode == "train":
            self.log_loss_metrics_path = os.path.join(HYPER_PARAMS.log_dir, mode, algo + "_" + policy)
            self.summary_writer_metrics_loss = SummaryWriter(self.log_loss_metrics_path)

        # TESTING MODE LOGGING SETUP
        if mode == "test":
            # Path for storing test results as CSV
            self.log_csv_test_path = os.path.join(
                HYPER_PARAMS.log_dir, f"{mode}_{extra}", algo + "_" + policy + "_" + time
            )
            os.makedirs(os.path.dirname(self.log_csv_test_path), exist_ok=True)

        # csv TRANSITION STORAGE SETUP
        if store_trs:
            if mode=="test":
                self.csv_transition_path = os.path.join(
                HYPER_PARAMS.agent_data_dir, f"{mode}_{extra}/transition_csv_{algo}_{policy}_{time}.csv"
            )
            else:
                self.csv_transition_path = os.path.join(
                HYPER_PARAMS.agent_data_dir, f"{mode}/transition_csv_{algo}_{policy}_{time}/{algo}_{policy}"
            )
            os.makedirs(os.path.dirname(self.csv_transition_path), exist_ok=True)

            self.transition_file_count = 0
            self.trans_idx_count = 0

    def log_info_test(self, info):
        """
        Append a single row of test episode metrics to a CSV file.
        Creates the CSV header if the file does not exist yet.

        Args:
            info (dict): Dictionary containing episode metrics to store.
        """
        file_exists = os.path.isfile(self.log_csv_test_path + ".csv")

        with open(self.log_csv_test_path + ".csv", "a") as f:
            csv_writer = DictWriter(
                f,
                delimiter=",",
                lineterminator="\n",
                fieldnames=[k for k in info],
            )  # Create a CSV writer object

            # If the log file doesn't exist, write the header
            if not file_exists:
                csv_writer.writeheader()

            #
            csv_writer.writerow(info)
            f.close()

    def store_trans_test(self,episode_count,obse, action, reward, done, new_obse):
        
        with open(self.csv_transition_path, mode="a", newline="") as f:
            # Write header
            csv_writer = csv.writer(
                f,
                delimiter=",",
                lineterminator="\n",
            )

            if os.path.getsize(self.csv_transition_path) == 0:
                obs_columns = [
                    "has_right_lane",
                    "has_left_lane",
                    "driving_in_weaving",
                    "dist_to_onramp",
                    "dist_to_offramp",
                    "leader_gap",
                    "leader_relatif_s",
                    "follower_gap",
                    "follower_relatif_s",
                    "left_leader_gap",
                    "left_leader_relatif_s",
                    "left_follower_gap",
                    "left_follower_relatif_s",
                    "right_leader_gap",
                    "right_leader_relatif_s",
                    "right_follower_gap",
                    "right_follower_relatif_s",
                ]
                header = ["trans_idx","episode_count"] + obs_columns + ["action", "reward", "done"] + [f"next_{col}" for col in obs_columns]

                # Write header
                csv_writer.writerow(header)

            # Write data rows
            csv_writer.writerow(
                [
                    self.trans_idx_count,
                    episode_count,
                    *obse,
                    action,
                    reward,
                    done,
                    *new_obse,
                ]
            )
            self.trans_idx_count += 1
            f.close()
    
    def store_trans_csv(self, obses, actions, rewards, dones, new_obses):
        file_name = self.csv_transition_path + "_" + str(self.transition_file_count) + ".csv"
        file_exists = os.path.isfile(file_name)

        if file_exists and os.path.getsize(file_name) / (1024 * 1024) > 90:
            self.transition_file_count += 1
            file_name = self.csv_transition_path + "_" + str(self.transition_file_count) + ".csv"
            file_exists = False

        with open(file_name, mode="a", newline="") as f:
            # Write header
            csv_writer = csv.writer(
                f,
                delimiter=",",
                lineterminator="\n",
            )

            if not file_exists:
                obs_columns = [
                    "has_right_lane",
                    "has_left_lane",
                    "driving_in_weaving",
                    "dist_to_onramp",
                    "dist_to_offramp",
                    "leader_gap",
                    "leader_relatif_s",
                    "follower_gap",
                    "follower_relatif_s",
                    "left_leader_gap",
                    "left_leader_relatif_s",
                    "left_follower_gap",
                    "left_follower_relatif_s",
                    "right_leader_gap",
                    "right_leader_relatif_s",
                    "right_follower_gap",
                    "right_follower_relatif_s",
                ]
                header = ["trans_idx"] + obs_columns + ["action", "reward", "done"] + [f"next_{col}" for col in obs_columns]

                # Write header
                csv_writer.writerow(header)

            for idx, item in enumerate(obses):
                # Write data rows
                csv_writer.writerow(
                    [
                        self.trans_idx_count,
                        *item.tolist(),
                        actions[idx],
                        rewards[idx],
                        dones[idx],
                        *new_obses[idx].tolist(),
                    ]
                )
                self.trans_idx_count += 1
            f.close()

    def store_obs_csv(self, episode_num, obs):
        file_name = self.csv_obs_path + "_" + str(self.obs_file_count) + ".csv"
        file_exists = os.path.isfile(file_name)

        if file_exists and os.path.getsize(file_name) / (1024 * 1024) > 90:
            self.obs_csv_count += 1
            file_name = self.csv_obs_path + "_" + str(self.obs_csv_count) + ".csv"
            file_exists = False

        with open(file_name, mode="a", newline="") as f:
            # Write header
            csv_writer = csv.writer(
                f,
                delimiter=",",
                lineterminator="\n",
            )

            if not file_exists:
                # Write header
                csv_writer.writerow(["index", "episode_num", "obs"])

            # Write data rows
            csv_writer.writerow([self.obs_idx_count, episode_num, json.dumps(obs.tolist())])
            self.obs_idx_count += 1
            f.close()

    def log_loss(self, tag, value, step):
        """
        Log a single scalar metric to TensorBoard.

        Args:
            tag (str): Name of the metric (used as the TensorBoard tag).
            value (float): Metric value to log.
            step (int): Training step (x-axis in TensorBoard).
        """
        self.summary_writer_metrics_loss.add_scalar(tag=tag, scalar_value=value, global_step=step)

    def log_episode_metrics(self, infos, dones):
        """
        Log episode-level metrics to TensorBoard for each completed episode.
        """

        for i, done in enumerate(dones):
            if done:
                episode_num = infos[i]["episode_num"]
                self.summary_writer_metrics_eps.add_scalar("episode_length", infos[i]["episode_length"], global_step=episode_num)
                self.summary_writer_metrics_eps.add_scalar(
                    "episode_total_reward", infos[i]["episode_total_reward"], global_step=episode_num
                )
                self.summary_writer_metrics_eps.add_scalar("episode_nb_lc", infos[i]["episode_nb_lc"], global_step=episode_num)
                self.summary_writer_metrics_eps.add_scalar(
                    "episode_nb_mingap_violation", infos[i]["episode_nb_mingap_violation"], global_step=episode_num
                )
                self.summary_writer_metrics_eps.add_scalar(
                    "episode_nb_emrgency_braking", infos[i]["episode_nb_emrgency_braking"], global_step=episode_num
                )
                self.summary_writer_metrics_eps.add_scalar(
                    "episode_nb_invalide_lc", infos[i]["episode_nb_invalide_lc"], global_step=episode_num
                )
                self.summary_writer_metrics_eps.add_scalar(
                    "episode_nb_vehicle_collision", infos[i]["episode_nb_vehicle_collision"], global_step=episode_num
                )
                self.summary_writer_metrics_eps.add_scalar(
                    "episode_nb_ping_pong", infos[i]["episode_nb_ping_pong"], global_step=episode_num
                )
                self.summary_writer_metrics_eps.add_scalar("episode_avg_speed", infos[i]["episode_avg_speed"], global_step=episode_num)
                self.summary_writer_metrics_eps.add_scalar("seed", infos[i]["seed"], global_step=episode_num)
