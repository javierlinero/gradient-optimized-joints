# install prereqs
fenics thru debian/linux:

must use v23.04 > of ubuntu bc mshr isn't a supported packaged in 22.04 LTS, make sure to run the following to fix:

```
sudo apt update && sudo apt dist-upgrade
sudo do-release-upgrade
```

or for a non-LTS 23.04  version run

```
sudo add-apt-repository "deb http://archive.ubuntu.com/ubuntu/ lunar main restricted universe multiverse"
```

After rebooting make sure to check the OS version with
```
lsb_release -a
```

Finally run the following to get fenics
```
sudo apt-get update
sudo apt-get install --no-install-recommends fenics
```

## Modules (thru python env)
A requirements.txt will be provided later on, but for now:
```
pip install git+https://github.com/dolfin-adjoint/pyadjoint.git
pip install dolfin-adjoint
pip install tqdm
pip install numpy
```

