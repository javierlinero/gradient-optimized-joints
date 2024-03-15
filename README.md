# install prereqs
Due to the Legacy Version in which the code was implemented (2021), we must use legacy FEniCS thru debian/linux setup:

[mshr](https://bitbucket.org/fenics-project/mshr/src/master/) is a package that isn't viable through v22.04 LTS, so we need v23.10:

```
sudo nano /etc/update-manager/release-upgrades
```

Set your prompt=normal and then do the following:

```
sudo apt update && sudo apt dist-upgrade
sudo do-release-upgrade
```

After installing the updated version, make sure to kill your terminal, restart WSL2 and check your version: 
```
lsb_release -a
```

Run the following to get fenics-legacy & gmsh:
```
sudo apt update && sudo apt dist-upgrade
sudo apt-get install --no-install-recommends fenics
sudo apt install python3-gmsh
```

## Modules (thru python env)
I have provided a requirements.txt file make sure to install all the proper packages, but if you'd like to do this manually use the following:

```
pip install pygmsh
pip install meshio
pip install git+https://github.com/dolfin-adjoint/pyadjoint.git
pip install dolfin-adjoint
pip install matplotlib
pip install tqdm
pip install torch
```

## Argsparser
Allows for the initial parameters of a mesh that needs to be optimized, where shapes.py has been augmented with numerous versions of interlocking joints. Likewise, it will also default to single dovetail joint for all program files.

## Setting up X11 Server for WSL2 Users
You can utilize any type of X server, but for the sake of making this simple the prereqs is installing ubuntu on wsl2 on a windows 11 computer. When launching VcXsrv (XLaunch) use the following settings:

1. Multiple Windows 
    1. (Display number=-1)
2. Start No client
3. Clipboard
    1. Primary Selection
4. Disable access control

Install the following python package
```
sudo apt-get install x11-apps
sudo apt-get install python3-tk
```
Now we need to edit the DISPLAY env name (add this to the very bottom of ~/.bashrc)
```
export DISPLAY=`grep -oP "(?<=nameserver ).+" /etc/resolv.conf`:0.0
```
Then make sure you update your terminal with
```
source ~/.bashrc
```

Now you can use any program and it will display it onto your local machine.

