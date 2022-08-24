The project name is: FPGA implementation of deep-learning guided MCTS (MLP part).

Abstract: The Monte Carlo Tree Search (MCTS) algorithm, which has achieved great success in many reinforcement learning (RL) applications, comprises four stages: selection, expansion, simulation, and backup. Interacting with the environment, workers simulate rewards following a policy based on either stochastic process functions or a trained Multilayer Perceptron (MLP). In this work, we implement an MLP inference accelerator using HLS treating its training as a black box, which can be easily customized to meet the requirements of throughput from upstream applications and the constraints of hardware resources. Experiments on popular RL networks show it has great scalability and is well-balanced between different stages.

* File:
  * results.xlsx: sheets summarizing the results of all experiments.
* File folders:
  * 1-4: Contains hardware synthesis results shown by vitis_analyzer for corresponding benchmarks. The file name matches the 'index' item in results.xlsx.
  * src: Contains typical source code and examples for these experiments. It's also explained how to change the code to get the customized implementation in readme.
  * history: Contains the intermediate versions before the final version used in experiments. There're also some documents helpful to understand the basic idea of the implementation.
  * tools: Contains the cmd to run the synthesis and a jupyter notebook to generate testbench data.

