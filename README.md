# PAIRS

This repositaty contains Torch code and trained models for the Paper:  A Unified Pose-aligned Representation for Fine-grained Recognition. The code is three parts: pose estimation network, patch feature extractors, and classification network.

## Pose Estimation Network

Self-defined FCN model: [LINK](http://google.com)

To run the model on test sets, run `th ./fcn.lua`

![FCN Architecture](https://i.imgur.com/FmkDkfS.png)

Stacked Hourglass model: [LINK](http://google.com)

## Patch Feature Extraction

To evaluate patch accuracy, run `th ./evaluate_patch.lua`

Extracted features: [LINK](http://google.com)

## Classification 

MLP model weights: [LINK](http://google.com)

To run the MLP model on the unified object representation, run `th mlp.lua`
