# PixFoundation-2.0: Do Video Multi-Modal LLMs Use Motion in Visual Grounding?
[Project Webpage]() [Paper](https://arxiv.org/pdf/2509.02807) [Datasets]()

Official implementation of my work PixFoundation 2.0. This work is part of a series of works that is trying to distill the current progress in pixel-level vision foundation models,
specifically the ones built using multi-modal large language models. The first work, [PixFoundation](https://arxiv.org/pdf/2502.04192), was focused on single images.

Code and Datasets to be released soon ...

## Motion-Centric Benchmark (MoCentric-Bench)

<div align="center">
<img src="https://github.com/MSiam/PixFoundation-2.0/blob/55545787a98f66348095697c17be8683211f3784/imgs/overview_1.png" width="70%" height="70%"><br><br>
</div>

### Benchmarking Pixel-level video MLLMs for referring video segmentation

<div align="center">
<img src="https://github.com/MSiam/PixFoundation-2.0/blob/55545787a98f66348095697c17be8683211f3784/imgs/overview_2.png" width="70%" height="70%"><br><br>
</div>

<div align="justify">
Multi-modal large language models (MLLMs) have shown impressive generalization across tasks using images and text modalities. While their extension to video has enabled tasks such as video question answering and video captioning, their pixel-level visual grounding abilities are less studied. In this work, we raise the pertinent question of whether motion is used in pixel-level visual grounding and whether video MLLMs can segment objects based on natural language expressions describing their motion patterns. We identify the shortcomings in the current benchmarks, where we show that a single frame can often suffice for capturing the motion referring expression without any temporal reasoning. To address this, we introduce four motion-centric probing techniques, particularly designed for the visual grounding task, to study video MLLMs’ ability to identify true motion from a fake one and their ability to grasp the motion order. Consequently, we provide a motion-centric benchmark, MoCentric-Bench. It ensures that video MLLMs are evaluated towards leveraging the interaction between motion and language rather than being dominated by static appearance cues emphasized in existing visual grounding datasets. We further establish strong single-image baselines that are on par with or outperform prior methods. Finally, we explore simple motion-centric adaptation techniques that provide state-of-the-art performance on our MoCentric-Bench. Our motion-centric benchmark, evaluation and findings challenge future models to improve dense spatiotemporal grounding and pixel-level understanding within videos.
</div>


### Benchmarking video MLLMs for region captioning
To be released soon ...

# References
Please cite my paper if you find it useful in your research

```
@article{siam2025pixfoundation-2.0,
  title={PixFoundation 2.0: Do Video Multi-Modal LLMs Use Motion in Visual Grounding?},
  author={Siam, Mennatullah},
  journal={arXiv preprint arXiv:2509.02807},
  year={2025}
}
```
