import tensorflow_datasets as tfds
import numpy as np
import os
from tqdm import tqdm

def _write_chunk(data_chunk, filename, first_chunk):
    """Append a chunk of data to an .npy file without loading the full file.

    The first time we write we overwrite the file (``first_chunk`` is True, mode
    'wb'); subsequent writes open the file in append mode ('ab'). Each call to
    ``np.save`` writes a complete NPY header + array. When reading, simply loop
    over ``np.load`` until EOF.
    """
    mode = "wb" if first_chunk else "ab"
    try:
        with open(filename, mode) as f:
            np.save(f, np.asarray(data_chunk))
    except ValueError:
        # Fallback for heterogeneous data (e.g. list with mixed dtypes/shapes)
        with open(filename, mode) as f:
            np.save(f, np.asarray(data_chunk, dtype=object))

def parse_stanford_hydra_dataset(step):
    obs = step['observation']
    action_vec = step['action']
    image = obs['image']
    instr = step['language_instruction'].decode()
    return image, instr, action_vec

def parse_viola_dataset(step):
    obs = step['observation']
    action = step['action']
    action_vec = np.concatenate([
        action['world_vector'],
        action['rotation_delta'],
        np.array([action['gripper_closedness_action']])
    ])
    image = obs['agentview_rgb']
    instr = obs['natural_language_instruction'].decode()
    return image, instr, action_vec

def convert_tf_to_numpy_dataset(dataset_dirs: list[str]):
    # Parameters controlling chunked writing
    CHUNK_SIZE = 50  # adjust to trade RAM vs. I/O overhead

    samples_chunk: list = []  # will store (image, instr, action_vec)
    raw_actions_chunk: list = []  # will store only action_vec

    # Track whether we've already written the first chunk so that we know
    # whether to overwrite or append.
    first_samples_chunk = True
    first_raw_chunk = True

    for tfrecord_dir in dataset_dirs:
        builder = tfds.builder_from_directory(tfrecord_dir)
        ds = builder.as_dataset(split="train")
        if "viola" in tfrecord_dir:
            parse_fn = parse_viola_dataset
        else:
            parse_fn = parse_stanford_hydra_dataset
        for sample in tqdm(tfds.as_numpy(ds), desc=f"Processing {tfrecord_dir}"):
            for step in sample['steps']:
                try:
                    image, instr, action_vec = parse_fn(step)
                    samples_chunk.append((image, instr, action_vec))
                    raw_actions_chunk.append(action_vec)

                    # Flush to disk if chunk is full to keep memory usage low
                    if len(samples_chunk) >= CHUNK_SIZE:
                        _write_chunk(samples_chunk, "data.npy", first_samples_chunk)
                        _write_chunk(raw_actions_chunk, "raw_actions.npy", first_raw_chunk)

                        # After the first write we switch to append mode
                        first_samples_chunk = False
                        first_raw_chunk = False

                        # Empty the in-memory buffers
                        samples_chunk.clear()
                        raw_actions_chunk.clear()
                except Exception as e:
                    print(f"Error processing sample in {tfrecord_dir}: {e}, {sample}")
                    continue

    # Flush anything that remains after the loop exits
    if samples_chunk:
        _write_chunk(samples_chunk, "data.npy", first_samples_chunk)
    if raw_actions_chunk:
        _write_chunk(raw_actions_chunk, "raw_actions.npy", first_raw_chunk)

convert_tf_to_numpy_dataset([
    "/workspace/viola/0.1.0",
    #"/workspace/stanford_hydra_dataset_converted_externally_to_rlds/0.1.0",
])