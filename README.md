
# GraspAgent_AnyHand
This repository is under construction. The final version will include a grasp policy learning for five hands, namely, Shadow hand five fingers, Shadow hand three fingers, Allegro hand, Robotica 2f85, and CasiaHand.


## Hands

### CasiaHand:
<div align="center">
  <img src="images/CH.png" alt="GraspAgent AnyHand Overview" width="800"/>
</div>

### ShadowHand five fingers
<div align="center">
  <img src="images/SH_5f.png" alt="GraspAgent AnyHand Overview" width="800"/>
</div>

### ShadowHand three fingers
<div align="center">
  <img src="images/SH.png" alt="GraspAgent AnyHand Overview" width="800"/>
</div>

### Allegro
<div align="center">
  <img src="images/allergo.png" alt="GraspAgent AnyHand Overview" width="800"/>
</div>

### Robotiq 3f85
<div align="center">
  <img src="images/2F85.png" alt="GraspAgent AnyHand Overview" width="800"/>
</div>

## Hand designs
Except for CasiaHand which was designed in our Lab, all hands are brought from the open source repository [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) with modification applied to each hand including changing the reference point and adding a mocap body.

## Prerequisites
The following pakages version are used during the development of this repository:
```
python=3.10.16
torch=2.5.1
open3d=0.18.0
cuda=12.6

```

## Final notes
This work is a result of intensive experiments conducted by Taqiaden during his work at Chinese Academy of Science Instititute of Automation. The result are promising and open the door for generalizing and automating the grasp policy for any given robotic hand. Setting up a new hand to the pipeline usually takes few minutes and training last 1 to 3 days using a single cuda gpu. For more details or any request to costumize the code for your specific hand design please do not hesitate to contact me: taqiaden@gmail.com  , whatsapp : 00967 774 631 499