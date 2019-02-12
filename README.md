# PAIRS

This repositaty contains the code and trained models for our WACV 2019 paper: "Aligned to the Object, not to the Image: A Unified Pose-aligned Representation for Fine-grained Recognition". It contains three major parts: 1. pose estimation network, 2. patch feature extractors, and 3. classification network. The later two parts are in the process of final cleaning and will be available soon.

## Pose Estimation Network

We have two sub-modules for pose estimation frametwork. One is written in Torch (Lua) with ResNet-50 as back-bone network. The other is written in Pytorch (Python) with ResNet-34 as back-bone network. 

## Patch Feature Extraction

Now the patch feature extractor network is added as Feature-Extraction folder. 

Torch needs to be installed in order to run the `simple.sh` for training.

Specify your own dataset path and pretrained model path in `simple.sh`.

<!---Self-defined FCN model: [LINK](http://google.com)-->

<!---To run the model on test sets, run `th ./fcn.lua`-->

<!---![FCN Architecture](https://i.imgur.com/FmkDkfS.png)-->



TODOS:


<!---To evaluate patch accuracy, run `th ./evaluate_patch.lua`-->

<!---Extracted features: [LINK](http://google.com) -->

## Classification 

<!---MLP model weights: [LINK](http://google.com)-->

<!---To run the MLP model on the unified object representation, run `th mlp.lua`-->
