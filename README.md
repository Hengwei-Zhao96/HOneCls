<h2 align="center">One-Class Risk Estimation for One-Class Hyperspectral Image Classification</h2>


<h5 align="right">
by <a href="https://hengwei-zhao96.github.io">Hengwei Zhao</a>,
<a href="http://rsidea.whu.edu.cn/">Yanfei Zhong</a>, 
and Xinyu Wang
</h5>

[[`Paper(ISPRS)`](https://www.sciencedirect.com/science/article/abs/pii/S0924271622000715)]
[[`Paper(TGRS)`](https://ieeexplore.ieee.org/document/10174705)]

---------------------

This is an official implementation of absNegative and One-class Risk Estimation in our ISPRS 2022 paper and TGRS 2023 paper, respectively.

## Applications:
1. Invasive tree species detection from airborne hyperspectral images [[`ISPRS (2022)`](https://www.sciencedirect.com/science/article/abs/pii/S0924271622000715)]
2. Pest and disease detection from UAV-borne hyperspectral images [[`JAG (2022)`](https://www.sciencedirect.com/science/article/pii/S1569843222001443)]
3. Crop extraction from spaceborne hyperspectral images [[`JAG (2021)`](https://www.sciencedirect.com/science/article/pii/S0303243421003056)]

## Requirements:
- pytorch >= 1.13.1
- GDAL ==3.4.1

## Running
1.Modify the data path in the configuration file (./configs/config_XX.py).
The hyperspectral data can be obtained from the [`Link`](https://pan.baidu.com/s/1Ac3ko3BcZ4sS_cmzZhA7ow?pwd=sqyy )(password:sqyy)

2.Training and testing
```bash
sh sh/HongHu.sh
sh sh/LongKou.sh
sh sh/HanChuan.sh
```

## Citation
If you use absNegative or One-class Risk Estimation in your research, please cite the following paper:
```text
@article{ZHAO2022328,
    title = {Mapping the distribution of invasive tree species using deep one-class classification in the tropical montane landscape of Kenya},
    journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
    volume = {187},
    pages = {328-344},
    year = {2022},
    issn = {0924-2716},
    doi = {https://doi.org/10.1016/j.isprsjprs.2022.03.005},
    url = {https://www.sciencedirect.com/science/article/pii/S0924271622000715},
    author = {Hengwei Zhao and Yanfei Zhong and Xinyu Wang and Xin Hu and Chang Luo and Mark Boitt and Rami Piiroinen and Liangpei Zhang and Janne Heiskanen and Petri Pellikka}
}

@ARTICLE{10174705,
    author={Zhao, Hengwei and Zhong, Yanfei and Wang, Xinyu and Shu, Hong},
    journal={IEEE Transactions on Geoscience and Remote Sensing}, 
    title={One-Class Risk Estimation for One-Class Hyperspectral Image Classification}, 
    year={2023},
    volume={},
    number={},
    pages={1-1},
    doi={10.1109/TGRS.2023.3292929}}
```
abgNegative and One-class Risk Estimation can be used for academic purposes only, and any commercial use is prohibited.
<a rel="license" href="https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en">

<img alt="知识共享许可协议" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>