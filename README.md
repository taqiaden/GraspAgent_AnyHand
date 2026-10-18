
# GraspAgent_AnyHand
Grasp policies for any hand. This framework learns grasp policies across heterogeneous robotic hands. Trained models for five hands are included, and getting a new hand ready to train takes roughly 15 minutes.

<p align="center">
    <a href='https://paper'>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper'>
    </a>
    <a href='https://huggingface.co/datasets/taqiaden/GraspAgent'>
      <img src='https://img.shields.io/badge/Dataset-Download-blue?style=plastic&logo=databricks&logoColor=white' alt='Dataset'>
    </a>
</p>

<div align="center">
  <img src="images/abstract_summary.jpg" alt="GraspAgent AnyHand Overview" width="800"/>
</div>

## Checkpoints and Datasets
Get the checkpoints and datasets from this [link](https://huggingface.co/datasets/taqiaden/GraspAgent). Paste and replace any existing file in this repo.

## Prerequisites
The following pakages version are used during the development of this repository:
```
python=3.10.16
torch=2.5.1
open3d=0.18.0
cuda=12.6
Mujoco=3.8.0

```

## Test mode

To evaluate a trained policy for a given hand, set `--test_mode true` in the corresponding script under the `training/` folder. The following arguments are supported:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--test_mode` | bool | `false` | Activate test mode. |
| `--view` | bool | `false` | Open the MuJoCo viewer to visualize the environment during testing. |
| `--steps` | int | `10` | Number of test steps to run. |

Example:

```bash
python training/<hand_script>.py --test_mode true --view true --steps 10
```

## Hand designs
Except for CasiaHand which was designed in our Lab, all hands are brought from the open source repository [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) with modification applied to each hand including changing the reference point and adding a mocap body.


## Final notes
Objects used in Mujoco simulator are pre-processed using the Collision-Aware Approximate Convex Decomposition (CoACD) method. The processing method is available [here](./sim_dexee/utils/generate_mesh_xml.py)

## Credits

This work was conceptualized and implemented by Taqiaden Alshameri during his research at the Institute of Automation, Chinese Academy of Sciences (CASIA).

Daheng Li contributed to the early establishment of the simulation environment and to solving design issues related to the CasiaHand.

Sizhe Wang collected the 500+ objects used in this work.