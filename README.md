<h1>
    Keypoints_RK3588
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
		<img src="https://github.com/Applied-Deep-Learning-Lab/Keypoints_RK3588/assets/109062816/b0af2540-a7ba-4c22-824e-39b4b9212886" width=350 alt="human" />
		<img src="https://github.com/Applied-Deep-Learning-Lab/Keypoints_RK3588/assets/109062816/65cacd91-1e83-4955-bd15-ce07586b2716" width=350 alt="kphuman" />
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
  git clone https://github.com/Applied-Deep-Learning-Lab/Keypoints_RK3588 
  cd Keypoints_RK3588
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
        4. Train keypoints model
        <img src="https://github.com/Applied-Deep-Learning-Lab/Keypoints_RK3588/assets/109062816/b7cb0f92-5084-47ab-876d-c66512e0b625" width=38 height=38 alt="Trainer" />
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
      ```

  * ### Create custom dataset

      You need create custom dataset for train your model.
      
      Step 1. Create a dataset directory like this:
      ```
      mmpose
	├── mmpose
	├── docs
	├── tests
	├── tools
	├── configs
	`── data
	    │── coco
		│-- annotations
		│   │-- train.json
		│   |-- val.json
		│   |-- test.json
		│-- train
		│   │-- 000000000009.jpg
		│   │-- 000000000025.jpg
		│   │-- 000000000030.jpg
		│   │-- ...
		│-- val
		│   │-- 000000000139.jpg
		│   │-- 000000000285.jpg
		│   │-- 000000000632.jpg
		│   │-- ...
		`-- test
		    │-- ...
      ```
      
      json annotation files should have a similar structure:
      ```
	{"licenses": [
	    {"name": "", 
	     "id": 0, 
	     "url": ""}], 
	 "info": {"contributor": "", 
		  "date_created": "", 
		  "description": "", 
		  "url": "", 
		  "version": "", 
		  "year": ""}, 
	 "categories": [
	     {"id": 1, 
	      "name": "pprofile", 
	      "supercategory": "", 
	      "keypoints": ["1", "2", "3", "4", 
		            ..., 
		            "29", "30", "31", "32"], 
	      "skeleton": [[25, 23], [3, 4], [26, 27], [16, 17], 
		           ..., 
		           [5, 29], [7, 8], [30, 31], [25, 26]]}, 
	 ], 
	 "images": [
	     {"id": 1, "width": 3510, "height": 2550, "file_name": "1.png", 
	      "license": 0, "flickr_url": "", "coco_url": "", "date_captured": 0}, 
	     ..., 
	     {"id": 7, "width": 3510, "height": 2550, "file_name": "7.png", 
	      "license": 0, "flickr_url": "", "coco_url": "", "date_captured": 0}], 
	"annotations": [
	    {"id": 1, 
	     "image_id": 1, 
	     "category_id": 1, 
	     "segmentation": [[1246.46, 1003.59, 1249.88, 1002.83, 1267.11, 1002.32,
		             ..., 
		             1236.96, 1017.27, 1239.87, 1011.57, 1242.41, 1005.87]], 
	     "area": 186851.55839999986, 
	     "bbox": [1237.94, 1002.38, 923.91, 202.24], 
	     "iscrowd": 0, 
	     "attributes": {"occluded": false}, 
	     "keypoints": [1240.94, 1008.88, 2, 1365.4, 1012.28, 2, 1484.1, 1010.12, 2, 
		          ...,
		          1772.73, 1058.51, 2, 1830.53, 1049.85, 2, 1942.12, 1062.11, 2],
	     "num_keypoints": 32},
	    {"id": 2, 
	     "image_id": 2, 
	     "category_id": 1, 
	     "segmentation": 
	     ...},
	     ...]}
      ```
      
      Step 2. Create config file for your data and goal.
      You can take and change the ready-made config:
      ```
      mim download mmpose --config associative_embedding_hrnet_w32_coco_512x512  --dest .
      ```
      You can make minimum changes in config for run training your model. 
      Change the following settings as needed:
      ```
      # Change the name of the dataset to the existing
      dataset_info['dataset_name']='BottomUpCocoDataset'

      # Change the information about keypoints and skeleton 
      # by analogy with the example as you need
      dataset_info['keypoint_info']=<keypoints_dict>
      dataset_info['skeleton_info']=<skeleton_dict>
      
      # Weights and sigmas is a "importance measure" of each key point. 
      # Feel free to experiment or set everything to 1.
      dataset_info['keypoint_info']=[1]*<num_keypoints>
      dataset_info['skeleton_info']=[1]*<num_keypoints>
      
      # Set information about your points in channel_cfg. 
      channel_cfg = dict(
          num_output_channels=<num_keypoints>,
          dataset_joints=<num_keypoints>,
          dataset_channel=[[<all_keypoints_ids>]],
          inference_channel=[<all_keypoints_ids>])
      
      # Set information about your points and input image size in data_cfg.
      data_cfg = dict(
		image_size=512,
		base_size=256,
		base_sigma=2,
		heatmap_size=[128],
		num_joints=<num_keypoints>,
		dataset_channel=[[<all_keypoints_ids>]],
		inference_channel=[<all_keypoints_ids>],
		...)
	
      # Set your parameters in model dict
      model['keypoint_head']['num_joints']=channel_cfg['dataset_joints']
      
      # Perhaps you should strip 'flip_index' from {val, test}_pipeline['meta_keys'] 
      # if your dataset doesn't have flip.
      
      # Specify the path to the directory with your dataset
      data_root = 'data/<dataset_folder>'
      
      # Set data dict
      data = dict(
		workers_per_gpu=2,
		train_dataloader=dict(samples_per_gpu=4),
		val_dataloader=dict(samples_per_gpu=1),
		test_dataloader=dict(samples_per_gpu=1),
	train=dict(
		type=dataset_info['dataset_name'],
		ann_file=f'{data_root}/train.json',
		img_prefix=f'{data_root}/train/',
		data_cfg=data_cfg,
		pipeline=train_pipeline,
		dataset_info=dataset_info),
	val=dict(
		type=dataset_info['dataset_name'],
		ann_file=f'{data_root}/val.json',
		img_prefix=f'{data_root}/val/',
		data_cfg=data_cfg,
		pipeline=val_pipeline,
		dataset_info=dataset_info),
	test=dict(
		type=dataset_info['dataset_name'],
		ann_file=f'{data_root}/test.json',
		img_prefix=f'{data_root}/test/',
		data_cfg=data_cfg,
		pipeline=test_pipeline,
		dataset_info=dataset_info))
      ```
      Step 3. Run training
      ```
      python tools/train.py <path/to/config.py>
      ```
      
      You can check your model:
      ```
      python demo/bottom_up_img_demo.py <path/to/config.py> \
      					<path/to/model.pth> \
      					--img-path <path/to/image.jpg> \
      					--out-img-root <path/to/results/folder>
      ```
</details>


<details>
  <summary>
    <h2>
      <p>
        5. Convert pytorch model to onnx to rknn
        <img src="https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Fds2converter.com%2Fwp-content%2Fuploads%2F2015%2F07%2Fconvert-icon.png&f=1&nofb=1&ipt=d6dbe833ced7274d7335d067ba819d63567e853dc093822f5cda0d18df3bfbdf&ipo=images" width=38 height=38 alt="Converter" />
      </p>
    </h2>
  </summary>

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
      git clone https://github.com/Applied-Deep-Learning-Lab/Keypoints_RK3588
      cd Keypoints_RK3588
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
