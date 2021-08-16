# Unsharp-Mask-Guided-Filtering

# Frequency-Supervised-MR-to-CT-Image-Synthesis
Unsharp Mask Guided Filtering, IEEE Transactions on Image Processing (TIP), 2021 [<a href="https://arxiv.org/pdf/2106.01428.pdf" target="_blank">pdf</a>]

![image](https://github.com/shizenglin/Unsharp-Mask-Guided-Filtering/blob/main/motivation.png)
<p> &#12288 &#12288 &#12288 &#12288 &#12288 &#12288 &#12288 &#12288 &#12288 Motivation of our approach </p>

![image](https://github.com/shizenglin/Unsharp-Mask-Guided-Filtering/blob/main/network.png)
<p> &#12288 &#12288 &#12288 &#12288 &#12288 &#12288 &#12288 &#12288 &#12288 Network architecture  of our approach </p>

<h2> Requirements </h2>
     1. CUDA 8.0 and Cudnn 7.5 or higher
<br> 2. GPU memory 10GB or higher
<br> 3. Python 2.7 or higher 
<br> 4. Tensorflow 2.0 or higher

<h2> Training </h2>
     1. Prepare your data (download the NYU Depth V2 dataset <a href="https://arxiv.org/pdf/2106.01428.pdf" target="_blank">here</a>) following Section V-A.
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

