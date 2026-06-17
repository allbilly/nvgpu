i want to understand how tinygrad NV backend works, by reimplemetation simple add in pure python
1. capture a sucess run
2. replay the hex blob with pure python with tinygrad helpers, but no runtime capture again everytime examples are run , make sure not to cheat with reusing stale vram
3. decode the hex blob in examples/*.py, this step is not completed if hex blob still appears
4. make each examples/*.py self contain, that is no tinygrad import except usb pcie related helpers
5. check is it better for examples/ to run ioctl submit

sample reference repo ask with deepwiki mcp
- allbilly/ane : examlpes/
- allbilly/rk3588 : examlpes/
- florianmattana/sass-king
- mikex86/LibreCuda
- cloudcores/CuAssembler
- vectorch-ai/ScaleLLM
- SzymonOzog/GPU_Programming
- gpuasm.com
- redplait/denvdis