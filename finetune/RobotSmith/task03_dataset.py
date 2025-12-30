import torch
import os
import numpy as np
from tqdm import tqdm
import time
import msgpack_numpy

msgpack_numpy.patch()
import copy


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        device,
        cameras=["front", "left_shoulder", "right_shoulder", "wrist"],
        ep_per_task=1000,
    ):
        self.device = device
        self.data_path = data_path
        self.train_data = []
        self.cameras = cameras
        time.sleep(5)

        self.train_tools = [18]  # TODO: put this in config
        self.construct_dataset(ep_per_task)

    def construct_dataset(self, ep_per_task):
        instruction = "flatten the dough to a height smaller than 0.03"
        self.num_tasks = 1  # TODO
        self.num_task_paths = 0

        for tool in tqdm(os.listdir(self.data_path)):
            if int(tool) not in self.train_tools:
                continue

            print("Loading tool:", tool)
            tool_dir = os.path.join(self.data_path, str(tool))

            for n_ep, ep in enumerate(os.listdir(tool_dir)):
                if n_ep >= ep_per_task:
                    break

                assert ep.endswith(".npz")
                episode = np.load(os.path.join(tool_dir, ep))

                # Filter for successful episodes: mean dough height less than 0.03
                if episode["penalty_mean"] > 0.03:
                    print("Skipping episode:", ep)
                    continue

                pcd = episode["pcd"]
                rgb = episode["rgb"]
                gripper_pose = episode["gripper_pose"]
                ignore_collisions = episode["ignore_collisions"]

                self.num_task_paths += 1
                num_steps = pcd.shape[0]

                for i in range(num_steps - 1):
                    sample = {}

                    for cam_idx, cam in enumerate(self.cameras):
                        sample[cam] = {
                            "pcd": np.transpose(pcd[i][cam_idx], (2, 0, 1)),
                            "rgb": np.transpose(rgb[i][cam_idx], (2, 0, 1)),
                        }

                    sample["gripper_pose"] = gripper_pose[i + 1]
                    time = (1.0 - (i / float(num_steps - 1))) * 2.0 - 1.0
                    sample["low_dim_state"] = np.concatenate(
                        [sample["gripper_pose"], [time]]
                    ).astype(np.float32)

                    sample["ignore_collisions"] = ignore_collisions[i + 1]
                    sample["lang_goal"] = instruction
                    sample["tasks"] = "task03_flatten"
                    self.train_data.append(copy.deepcopy(sample))

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.train_data[idx]
        # TODO: later
        # sample["lang_goal"] = random.choice(
        #     sample["lang_goal"]
        # )  # randomly choose one instruction for every fetching. This is important for generalization.


if __name__ == "__main__":
    dataset = Dataset(
        data_path="/home/amli/research/RobotSmith/task03_flatten_tools/episodes",
        device="cuda:0",
        ep_per_task=100,
    )

    print(len(dataset))

    for data in dataset:
        print(data.keys())
        print(data["lang_goal"])
        break
