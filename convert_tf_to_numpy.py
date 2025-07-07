import h5py
import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm


def write_datafile(
    filename: str, images: list[np.ndarray], actions: list[np.ndarray], instrs: list[str]
):
    with h5py.File(filename, "w") as f:
        f.create_dataset("images", data=images)
        f.create_dataset("actions", data=actions)
        string_dtype = h5py.special_dtype(vlen=str)
        f.create_dataset("instrs", data=instrs, dtype=string_dtype)


def parse_stanford_hydra_dataset(step):
    obs = step["observation"]
    action_vec = step["action"]
    image = obs["image"]
    instr = step["language_instruction"].decode()
    return image, instr, action_vec


def parse_viola_dataset(step):
    obs = step["observation"]
    action = step["action"]
    action_vec = np.concatenate(
        [
            action["world_vector"],
            action["rotation_delta"],
            np.array([action["gripper_closedness_action"]]),
        ]
    )
    image = obs["agentview_rgb"]
    instr = obs["natural_language_instruction"].decode()
    return image, instr, action_vec


def convert_tf_to_numpy_dataset(dataset_dirs: list[str]):
    CHUNK_SIZE = 30000  # adjust to trade RAM vs. I/O overhead

    images = []
    actions = []
    instrs = []
    chunk_idx = 0

    def reset_trajectory():
        nonlocal images
        nonlocal actions
        nonlocal instrs
        nonlocal chunk_idx
        images = []
        actions = []
        instrs = []
        chunk_idx += 1

    for tfrecord_dir in dataset_dirs:
        print(f"Processing {tfrecord_dir}")
        dataset_name = tfrecord_dir.split("/")[1]
        builder = tfds.builder_from_directory(tfrecord_dir)
        ds = builder.as_dataset(split="train")
        print(f"Found {len(ds)} samples")
        if "viola" in tfrecord_dir:
            parse_fn = parse_viola_dataset
        else:
            parse_fn = parse_stanford_hydra_dataset
        for sample in tqdm(tfds.as_numpy(ds), desc=f"Processing {tfrecord_dir}"):
            for step in sample["steps"]:
                try:
                    image, instr, action_vec = parse_fn(step)
                    images.append(image)
                    actions.append(action_vec)
                    instrs.append(instr)
                    if len(actions) >= CHUNK_SIZE:
                        write_datafile(
                            f"data/{dataset_name}_{chunk_idx}.h5", images, actions, instrs
                        )
                        reset_trajectory()
                except Exception as e:
                    print(f"Error processing sample in {tfrecord_dir}: {e}")
                    continue

        # Final chunk for this dataset
        if len(actions) > 0:
            write_datafile(f"data/{dataset_name}_{chunk_idx}.h5", images, actions, instrs)
        reset_trajectory()


convert_tf_to_numpy_dataset(
    [
        "data/viola/0.1.0",
        # "/workspace/stanford_hydra_dataset_converted_externally_to_rlds/0.1.0",
    ]
)
