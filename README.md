<h1>
    Keypoints_HRNet_RK3588
</h1>

<details open>
  <summary>
    <h2>
      <p>
        Abstract
      </p>
    </h2>
  </summary>   
	<b>We provide a solution for launching Keypoints Search (Pose Estimation) neural networks on RK3588.<br>
	<b>The process for preparing the edge device is described below.<br>
	<b>We also provide a quick guide to training and converting models.<br>
	<b><br>  	
	Example:
	<div>
		<img src="https://github.com/Applied-Deep-Learning-Lab/Keypoints_HRNet_RK3588/assets/109062816/b0af2540-a7ba-4c22-824e-39b4b9212886" width=350 alt="human" />
		<img src="https://github.com/Applied-Deep-Learning-Lab/Keypoints_HRNet_RK3588/assets/109062816/65cacd91-1e83-4955-bd15-ce07586b2716" width=350 alt="kphuman" />
	</div>
		
</details>

<details open>
  <summary>
    <h2>
      <p>
        1. Prerequisites
        <img src="https://www.svgrepo.com/show/288488/motherboard.svg" width=38 height=38 alt="Prerequisites" />
      </p>
    </h2>
  </summary>   
  
  * ### Ubuntu

    Install Ubuntu on your RK3588 device. *(tested on Ubuntu 20.04 and OrangePi5/Firefly ROC RK3588S devices)*

    For installing Ubuntu on Firefly you can use their manual[[1]](https://wiki.t-firefly.com/en/ROC-RK3588S-PC/index.html)[[2]](https://en.t-firefly.com/doc/download/page/id/142.html).

    For installing Ubuntu on OrangePi you can use [their manual](http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/service-and-support/Orange-pi-5.html).

    Or use ours **README's** for them *(select the one below)*.

    |[OrangePi](https://github.com/Applied-Deep-Learning-Lab/Yolov5_RK3588/blob/main/resources/OrangePi/README_ORANGEPI.md)|[Firefly](https://github.com/Applied-Deep-Learning-Lab/Yolov5_RK3588/blob/main/resources/Firefly/README_FIREFLY.md)|
    |                 :---:                 |                :---:               |
</details>

<details open>
  <summary>
    <h2>
      <p>
        2. Installing and configurating
        <img src="https://cdn1.iconfinder.com/data/icons/user-interface-cute-vol-2/52/configuration__settings__options__config-512.png" width=38 height=38 alt="Configurations" />
      </p>
    </h2>
  </summary>

  Install miniconda

  ```
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
  bash Miniconda3-latest-Linux-aarch64.sh
  ```

  Then rerun terminal session:

  ```
  source ~/.bashrc
  ```

  Create conda env with python3.8
  ```
  conda create -n <env-name> python=3.8
  conda activate <env-name>
  ```

  Clone repository:
  ```
  git clone https://github.com/Applied-Deep-Learning-Lab/Keypoints_HRNet_RK3588 
  cd Keypoints_HRNet_RK3588
  ```

  Install RKNN-Toolkit2-Lite

  ```
  pip install install/rknn_toolkit_lite2-1.5.0-cp38-cp38-linux_aarch64.whl
  ```

  In created conda enviroment also install requirements from the same directory

  ```
  pip install -r install/requirements.txt
  ```

</details>

<details open>
  <summary>
    <h2>
      <p>
        3. Running the keypoints search
        <img src="https://cdn1.iconfinder.com/data/icons/pain/154/body-health-shock-dots-pain-man-512.png" width=38 height=38 alt="Keypoints" />
      </p>
    </h2>
  </summary>

  ``main.py`` runs inference like:
  
  ```
  python3 main.py weights/human_pose.rknn \
                  images/human.jpg
  ```

  Inference results are saved to the ./results folder by default.

</details>


<details>
  <summary>
    <h2>
      <p>
        4. Convert pytorch model to onnx to rknn
        <img src="https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Fds2converter.com%2Fwp-content%2Fuploads%2F2015%2F07%2Fconvert-icon.png&f=1&nofb=1&ipt=d6dbe833ced7274d7335d067ba819d63567e853dc093822f5cda0d18df3bfbdf&ipo=images" width=38 height=38 alt="Converter" />
      </p>
    </h2>
  </summary>
	
  * ### Preparation Host PC

      For training model we use MMPose by OpenMMLab.
      
      Step 0. You will also need conda on the host PC.

      ```
      conda create -n openmmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
      conda activate openmmlab
      ```

      Step 1. Install MMCV using MIM.
      ```
      sudo apt-get update
      pip3 install -U openmim
      mim install mmcv-full==1.7.0
      ```
      
      Step 2. Install MMPose.
      ```
      git clone --depth 1 --branch v0.29.0 https://github.com/open-mmlab/mmpose.git
      cd mmpose
      pip install -r requirements.txt
      pip install -v -e .
      pip install numpy==1.23.5
      ```

  * ### Convert pytorch to onnx
	

      Inside mmpose folder and conda 'openmmlab' environment:
      ```
      python tools/deployment/pytorch2onnx.py <path/to/config.py> \
						<path/to/model.pth> \
						--output-file <path/to/model.onnx> \
						--shape 1 3 <model_size> <model_size>
      ```

  * ### Convert onnx to rknn

      Step 1. Create conda environment
      ```
      conda create -n rknn python=3.8
      conda activate rknn
      ```
      
      Step 2. Install RKNN-Toolkit2
      ```
      git clone https://github.com/Applied-Deep-Learning-Lab/Keypoints_HRNet_RK3588
      cd Keypoints_HRNet_RK3588
      pip install install/rknn_toolkit2-1.5.0+1fa95b5c-cp38-cp38-linux_x86_64.whl
      ```

      Step 3. For convert your *.onnx* model to *.rknn* run **onnx2rknn.py** like:
      ```
      python onnx2rknn.py <path/to/model.onnx>
      
      # For more precise conversion settings, 
      # check the additional options in the help:
      # python onnx2rknn.py -h
      ```
      

</details>
