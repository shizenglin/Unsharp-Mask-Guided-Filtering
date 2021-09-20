# [Unsharp Mask Guided Filtering, IEEE Transactions on Image Processing (TIP), 2021](https://arxiv.org/pdf/2106.01428.pdf)

![image](https://github.com/shizenglin/Unsharp-Mask-Guided-Filtering/blob/main/motivation.png)

![image](https://github.com/shizenglin/Unsharp-Mask-Guided-Filtering/blob/main/network.png)

<h1> Unsharp-mask guided filtering without learning </h1>
<br> You can find the code in the folder of "matlab"

<h2> Usage </h2>
<br> Run "example_smoothing.m", "example_enhancement.m", and "example_flash.m" to get the Figures 3, 4 and 5

<h1> Unsharp-mask guided filtering with learning </h1>
<br> You can find the code in the folder of "learning"

<h2> Requirements </h2>
     1. CUDA 8.0 and Cudnn 7.5 or higher
<br> 2. GPU memory 10GB or higher
<br> 3. Python 2.7 or higher 
<br> 4. Tensorflow 2.0 or higher. If your Tensorflow version is lower than 2.0, you should replace "import tensorflow.compat.v1 as tf" with "import tensorflow as tf" in "main.py", "model.py", and "ops.py"

<h2> Training </h2>
     1. Prepare your data (download the NYU Depth V2 dataset <a href="https://drive.google.com/file/d/1RAYK7zm_qXp6nrzjaNVaBRQc8sk9hzkn/view?usp=sharing" target="_blank">here</a>) following Section V-A.
<br> 2. Set the experiment settings in ¨tr_param.ini¨ in which phase = train, and set other parameters accordingly (refer to our paper).
<br> 3. Run ¨python main.py¨

<h2> Testing </h2>
     1. Prepare your data following Section V-A.
<br> 2. Set the experiment settings in ¨tr_param.ini¨ in which phase = test, and set other parameters accordingly (refer to our paper).
<br> 3. Run ¨python main.py¨


Please cite our paper when you use this code.

     @inproceedings{shi2021filtering,
     title={Unsharp Mask Guided Filtering},
     author={Zenglin Shi, Yunlu Chen, Efstratios Gavves, Pascal Mettes, and Cees G. M. Snoek},
     booktitle={IEEE Transactions on Image Processing},
     year={2021}
     }

