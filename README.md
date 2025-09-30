# Preparation

Download two gazebo models to your local model folder first.

1. Link: https://app.gazebosim.org/ctrazziwp/fuel/models/pallet
2. Link: https://app.gazebosim.org/QCforward/fuel/models/carboard%20box
3. Move to `~/.gazebo/models`

PS: cardboard_box folder should be renamed manually, from `carboard box` to `cardboard_box`, otherwise gazebo spawn may fail. I encountered this issue on my laptop.

# Launch env

`roslaunch pallet_detection pallet_detection.launch`
