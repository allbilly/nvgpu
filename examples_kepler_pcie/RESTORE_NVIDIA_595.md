# Restore RTX 3080 Ti stack after Kepler 470 test

```sh
sudo apt install -y nvidia-driver-595-open
sudo reboot
```

Confirm: `nvidia-smi` shows GeForce RTX 3080 Ti; `09:00.0` stays unbound (Kepler unsupported on 595).
