<div align="center">
    <b><font size="5">Aligning Large Language Models from Self-Reference AI Feedback</font></b><br>
    <b><font size="5">with one General Principle</font></b>
    <sup>
    </sup>
    <div> </div>
</div>


<div align="center">

[📘License](#license) |
[🤗Dataset](https://huggingface.co/datasets/rbao2018/Self_Ref_Feedback) |
[📜Paper](https://arxiv.org/abs/2406.11190)

</div>

## Introduction
Our project has implemented the training process of the self-reference AI feedback with one general principle. We also partially refactoring the [OpenRLHF](https://github.com/OpenLLMAI/OpenRLHF) framework to improve the efficiency of the PPO algorithm.
## Engineering Techniques

### Model Deployment
- Utilize [lmdeploy](https://github.com/InternLM/lmdeploy) for deploying models, enabling quick access to AI feedback and model generation.
## Usage Instructions

### Prerequisites
1. Install [Vllm](https://github.com/vllm-project/vllm) version 0.4.1 or higher. This library typically reinstall a newer version of PyTorch.
2. Install the corresponding version of [OpenRLHF](https://github.com/OpenLLMAI/OpenRLHF).

### PPO Algorithm Efficiency Improvements
1. Replace the original Deepspeed framework with the FSDP framework to reduce GPU memory usage and increase training speed.
2. Optimize the scheduling algorithm for asynchronous actor-critic training in the PPO training process to enhance overall framework efficiency.
3. Improve the implementation of experience replay generation to avoid the inefficiency of multiple small-batch reply generations by Vllm.


## License

The code is licensed under Apache-2.0, while model weights are fully open for academic research.

## References & Acknowledgements

We would like to express our gratitude to the following projects and organizations for their contributions to the field of generative AI:

- [Vllm](https://github.com/vllm-project/vllm)
- [OpenRLHF](https://github.com/OpenLLMAI/OpenRLHF)
- [lmdeploy](https://github.com/InternLM/lmdeploy)
