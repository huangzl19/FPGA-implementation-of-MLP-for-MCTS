There're 4 typical source codes for each net; by running them, you can get results in sub-sheet '4 in 1'.

For experiments with other settings, they can be easily modified following these rules:

* If you want to change to a bigger $BSIZE$ but leave the others the same, just modify `BSIZE` in  `Mydataflow.h`
* If you want to change $k$ (the resources used), should first recalculate the $P_i$ for all layers in the network, and then decide which template to use ($P_i < L_{i+1}$ or $P_i ≥ L_{i+1}$). Finally, apply the corresponding implementation to the code. Note that code examples for $P_i < L_{i+1}$ and $P_i ≥ L_{i+1}$ are available in `./CartPole-B1024-K4/Mydataflow.cpp`, function `blockMul1` and `unrollMul2` respectively. Generally, constant numbers in `Mydataflow.h` and unroll factors in `Mydataflow.cpp` should be modified.
* If you want to change the whole network architecture, you should recalculate $BN$ and $P_i$ for the whole network and modify the code example, just like above. Note that the number of layers may change, so additional MM and activation layers should be added. However, essentially there is no difference from the example given. Both `Mydataflow.h` and `Mydataflow.cpp` should be revised.
* You can change many factors by changing them one by one and then composing them together.

The synthesis reports, bit files, source codes for other experiments and so on are available on Yangzi, `/home/zongle/myTest/`.

