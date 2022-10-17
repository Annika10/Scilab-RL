#!/bin/bash

setup_venv() {
  # Check if there is already a venv
  if [ -d "venv" ]; then
    echo "Skipping venv setup as there is already a venv."
    return
  fi
  # check if venv directory is already present
  if [ ! -d "$PWD/venv" ]; then
    # Set up the venv
    echo "Setting up venv"
    if [ ! -x "$(command -v virtualenv )" ]; then
      virtualenv -p python3 venv;
    else
      python3 -m venv venv;
    fi
  fi
  source venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
}

setup_conda() {
  printf "Setup conda"
	source $(conda info --base)/etc/profile.d/conda.sh
  # check if scilabrl already exists
	if [ conda env list | grep "scilabrl*" >/dev/null 2>&1 ]; then
    conda activate scilabrl
    pip install -r requirements.txt
  else
    if [ $(uname -s) == "Linux" ]; then
      conda env create -f conda/linux_environment.yaml
      conda install cudatoolkit=11.3 pytorch -c pytorch -y
    elif [ $(uname -s) == "Darwin" ]; then
      if [ $(uname -m) =~ "arm" ]; then
        conda env create -f conda/macos_arm_environment.yaml
      elif [ $(uname -m) =~ "x86" ]; then
        printf "Intel Macs are currently not supported"
        exit 1
        # conda env create -f macos_x86_environment.yaml
      fi
    fi
    conda activate scilabrl
  fi
}

get_mujoco() {
  # Check if MuJoCo is already installed
  if [ -d "${HOME}/.mujoco/mujoco210" ]; then
    echo "Skipping MuJoCo as it is already installed."
    return
  fi
  # Get MuJoCo
  echo "Getting MuJoCo"
  MUJOCO_VERSION="2.1.1"
  if [ $(uname -s) == "Linux" ]; then
    MUJOCO_DISTRO="linux-x86_64.tar.gz"
    wget "https://github.com/deepmind/mujoco/releases/download/$MUJOCO_VERSION/mujoco-$MUJOCO_VERSION-linux-x86_64.tar.gz" -O "${HOME}/mujoco.tar.gz"
    tar -xf "${HOME}/mujoco.tar.gz" -C "${HOME}/.mujoco/mujoco210"
    rm "${HOME}/mujoco.tar.gz"
  elif [ $(uname -s) == "Darwin" ]; then
    MUJOCO_DISTRO="macos-universal2.dmg"
    wget "https://github.com/deepmind/mujoco/releases/download/$MUJOCO_VERSION/mujoco-$MUJOCO_VERSION-macos-universal2.dmg"
    VOLUME=`hdiutil attach mujoco-2.1.1-macos-universal2.dmg | grep Volumes | awk '{print $3}'`
    cp -rf $VOLUME/*.app /Applications
    hdiutil detach $VOLUME
    mkdir -p $HOME/.mujoco/mujoco210
    ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/Headers/ $HOME/.mujoco/mujoco210/include
    mkdir -p $HOME/.mujoco/mujoco210/bin
    ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/libmujoco.2.1.1.dylib $HOME/.mujoco/mujoco210/bin/libmujoco210.dylib
    ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/libmujoco.2.1.1.dylib /usr/local/lib/
    # For M1 (arm64) mac users:
    if [ $(uname -m) =~ "arm" ]; then
      conda install -y glfw
      rm -rfiv $CONDA_PREFIX/lib/python3.*/site-packages/glfw/libglfw.3.dylib
      ln -sf $CONDA_PREFIX/lib/libglfw.3.dylib $HOME/.mujoco/mujoco210/bin
      if [ ! -x "$(command -v gcc-12)" ]; then
        brew install gcc
      fi
      export CC=/opt/homebrew/bin/gcc-12
    fi
    rm -rf mujoco-2.1.1-macos-universal2.dmg
  fi
  # Install mujoco-py
  echo "Installing mujoco-py and testing import"
  source set_paths.sh
  pip install mujoco-py && python -c 'import mujoco_py'
}

get_rlbench() {
  if [ $(uname -s) == "Darwin" ]; then
    echo "There is no PyRep support for macos"
    return
  fi
  # Check if CoppeliaSim is already installed
  if [ -d "${HOME}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04" ]; then
    echo "Skipping CoppeliaSim as it is already installed."
    return
  fi
  # Get CoppeliaSim
  echo "Getting CoppeliaSim"
  wget http://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -O "${HOME}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz"
  tar -xf "${HOME}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz" -C "$HOME"
  rm "${HOME}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz"
  # Get RLBench
  echo "Getting RLBench"
  source venv/bin/activate
  source set_paths.sh
  pip install git+https://github.com/stepjam/PyRep.git git+https://github.com/stepjam/RLBench.git pyquaternion natsort
}


# check if conda is available
if [ -x "$(command -v conda)" ]; then
  setup_conda
else
  setup_venv
fi

while getopts 'mr' OPTION; do
  case "$OPTION" in
    m)
      get_mujoco
      ;;
    r)
      get_rlbench
      ;;
    ?)
      echo "Use -m to install MuJoCo and -r to install RLBench"
      exit 1
      ;;
  esac
done
