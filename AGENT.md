i want to understand how tinygrad NV backend works, by reimplemetation simple add in pure python
1. capture a sucess run
2. replay the hex blob with pure python with tinygrad helpers, but no runtime capture again everytime examples are run , make sure not to cheat with reusing stale vram
3. decode the hex blob in examples/*.py
4. make each examples/*.py self contain, that is no tinygrad import except usb pcie related helpers

sample reference deepwiki
- allbilly/ane : examlpes/
- allbilly/rk3588 : examlpes/