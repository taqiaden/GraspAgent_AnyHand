
# GraspAgent_AnyHand
**GraspAgent_AnyHand** implements an adversarial learning framework for training grasp policies across heterogeneous robotic hands. It includes pretrained models for five hands and supports configuring a new hand for training in approximately 15 minutes.

<div align="center">
  <img src="images/abstract_summary.jpg" alt="GraspAgent AnyHand Overview" width="800"/>
</div>
(A) The framework takes a robot hand as input. (B) The proposed learned components include neural networks used at inference (Sampler, safety and quality networks), and a critic network used only for training. (C) The engineering part for a given hand includes setting up the simulator and defining heuristic rules. (D) We propose an adversarial learning mechanism. (E) The framework outputs a grasp policy and a robust grasp pose dataset for the hand.

## Supported Robotic Hands

The repository includes trained models for five heterogeneous robotic hands:

- Shadow Dexterous Hand
- DEX-EE (Shadow three-finger hand)
- CasiaHand
- Allegro Hand V4
- Robotiq 2F85

## Checkpoints and Datasets
Get the checkpoints and datasets from this [link](https://anonymous-hf.com/a/nlm8uiuqaius/). Copy all downloaded folders into the root of this repository (the same folder as this README.md), overwriting any existing files when prompted.

## Prerequisites
The following software versions were used during development:

| Package | Version |
|---|---|
| Python | 3.10.16 |
| PyTorch | 2.5.1 |
| Open3D | 0.18.0 |
| CUDA | 12.6 |
| MuJoCo | 3.8.0 |

A compatible NVIDIA GPU and CUDA environment are required for GPU-accelerated training. Ensure that your installed PyTorch build and NVIDIA driver are compatible with your CUDA setup.

## Evaluating a Trained Policy

To evaluate a pretrained policy, run the corresponding script in the `training/` directory with test mode enabled.

### Command-line Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--test_mode` | bool | `false` | Enable evaluation mode. |
| `--view` | bool | `false` | Open the MuJoCo viewer during evaluation. |
| `--steps` | int | `10` | Number of evaluation steps. |

### Example

Replace `<hand_script>` with the appropriate script for your target hand.

```bash
python training/<hand_script>.py --test_mode true --view true --steps 10
```

This command enables visualization and runs 10 evaluation steps.
## Hand designs
Except for CasiaHand which was designed in our Lab, all hands are brought from the open source repository [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) with modification applied to each hand including changing the reference point and adding a mocap body.


## Final notes
Objects used in Mujoco simulator are pre-processed using the Collision-Aware Approximate Convex Decomposition (CoACD) method. The processing method is available [here](./sim_dexee/utils/generate_mesh_xml.py)

## Credits

This work was conceptualized and implemented by **Taqiaden Alshameri** during his research at the Peng Wang Laboratory of Robot Learning and Intelligent Systems, Institute of Automation, Chinese Academy of Sciences.

- **Daheng Li** contributed to the early establishment of the simulation environment and to solving design issues related to the CasiaHand.

- **Sizhe Wang** collected the 500+ objects used in this work.