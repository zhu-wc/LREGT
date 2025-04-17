# LREGT: Local Relationship Enhanced Gated Transformer for Image Captioning

This repository contains the reference code for the paper **LREGT: Local Relationship Enhanced Gated Transformer for Image Captioning**

## Experiment setup

Most of the previous works follow [m2 transformer](https://github.com/aimagelab/meshed-memory-transformer), but they utilized some lower-version packages. Therefore, we recommend  referring to [Xmodal-Ctx](https://github.com/GT-RIPL/Xmodal-Ctx). 

## Data preparation

* **Annotation**. Download the annotation file [annotation.zip](https://drive.google.com/file/d/1i8mqKFKhqvBr8kEp3DbIh9-9UNAfKGmE/view?usp=sharing). Extarct and put it in the project root directory.
* **Feature**. We extract feature with the code in [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa). You can download the features we used [here](https://github.com/luo3300612/image-captioning-DLCT).
* **evaluation**. We use standard evaluation tools to measure the performance of the model, and you can also obtain it [here](https://github.com/luo3300612/image-captioning-DLCT). Extarct and put it in the project root directory.

## Training

```python
python train.py --devices 0
```

If you wish to recurrent our code, you need modify some parameters in `train.py`, such as `--features_path` and `--annotation_floder`.

## References

[1] [M2](https://github.com/aimagelab/meshed-memory-transformer)

[2] [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa)

[3] [Xmodal-Ctx](https://github.com/GT-RIPL/Xmodal-Ctx)
