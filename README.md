
<div align="center">
  
## Bidirectional Translation between UHD-HDR and HD-SDR Videos [Paper](https://ieeexplore.ieee.org/document/10025794/)

[Mingde Yao](https://scholar.google.com/citations?user=fsE3MzwAAAAJ&hl=en), [Dongliang He](https://scholar.google.com/citations?user=ui6DYGoAAAAJ&hl=en), [Xin Li](https://scholar.google.com/citations?user=4BEGYMwAAAAJ&hl=zh-CN), [Zhihong Pan](https://scholar.google.com/citations?user=IVxQvz0AAAAJ&hl=en), and [Zhiwei Xiong](http://staff.ustc.edu.cn/~zwxiong/)*

*Corresponding Author, University of Science and Technology of China (USTC)

:rocket: This is the official repository of HDR-BiTNet (TMM 2023). 

</div>

**HDR-BiTNet aims at addressing the practical translation between UHD-HDR (HDRTV) and HD-SDR (SDRTV) videos.**

We provide the training and test code along with the trained weights and the dataset (train+test) used for the HDR-BiTNet. 

:heart: If you find this repository useful, please star this repo :star2: and cite our paper :page_facing_up:.

**Reference**:  

> Mingde Yao, Dongliang He, Xin Li, Zhihong Pan, and Zhiwei Xiong, "Bidirectional Translation Between UHD-HDR and HD-SDR Videos",
*IEEE Transactions on Multimedia*, 2023.

**Bibtex**:

```
@article{yao2023bidirectional,
  title={Bidirectional Translation Between UHD-HDR and HD-SDR Videos},
  author={Yao, Mingde and He, Dongliang and Li, Xin and Pan, Zhihong and Xiong, Zhiwei},
  journal={IEEE Transactions on Multimedia},
  year={2023},
  publisher={IEEE}
}
```

## Video example

![test_set_AdobeExpress_AdobeExpress_AdobeExpress](https://github.com/mdyao/HDR-BiTNet/assets/33108887/dc5b1ebe-de24-444f-ab5e-818bed782ea4)

This GIF image has been tone-mapped to SDR and compressed for visibility.

See more comparison of SDR video and HDR video in `example/` folder. 

:warning: **Note**: The `HDR.mp4` file is encoded to comply with the HDR10 standard. For the best viewing experience, please watch the video on [certified HDR displays](https://displayhdr.org/certified-products) with maximum brightness. Alternatively, a Mac Book Pro can be used for convenience.

## Test code

### Quick Start
1. Download the test dataset from [this link](https://drive.google.com/open?id=144QYC403NrFXunlsr4k8MXUCxrlauVYH).
2. Unzip and place the 'test' dataset in a proper folder, _e.g.,_ `/testdata`.
3. Put the pretrained model file (./model.pth) in a proper folder.
4. Set a config file in options/test/, then run as following:

 ```
 python test.py -opt options/test/test.yml
 ```


## Code Framework
The code framework follows [BasicSR](https://github.com/xinntao/BasicSR/tree/master/codes). 

### Contents

**Config**: [`options/`](./options) Configure the options for data loader, network structure, model, training strategies and etc.

**Data**: [`data/`](./data) A data loader to provide data for training, validation and testing.

**Model**: [`models/`](./models) Construct models for training and testing.

**Network**: [`models/modules/`](./models/modules) Construct network architectures.



<!-- This repository is the **official implementation** of the paper, "Bidirectional Translation Between UHD-HDR and HD-SDR Videos", where more implementation details are presented. -->

## Contact

If you have any problem with the released code, please do not hesitate to open an issue.

For any inquiries or questions, please contact me by email (mdyao@mail.ustc.edu.cn).
