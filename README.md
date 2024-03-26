# Dual-adversarial-network-for-cross-domain-open-set-fault-diagnosis

[RESS 2022] Dual adversarial network for cross-domain open set fault diagnosis


## Paper

Paper link: [Dual adversarial network for cross-domain open set fault diagnosis](https://www.sciencedirect.com/science/article/pii/S0951832022000370)

## Abstract

Recently, cross-domain fault diagnosis methods have been successfully developed and applied. Among them, the ones exhibiting the best performance rely on the common assumption that the training and testing data share an identical label space, implying that fault modes are the same in different engineering scenarios. However, fault modes in the testing phase are unpredictable and new fault modes usually occur, posing challenges for existing cross-domain methods regarding their effectiveness. To address such challenges, a novel open set domain adaptation network based on dual adversarial learning is proposed in this study. An auxiliary domain discriminator assigns similarity weights for individual target samples to distinguish between known and unknown fault modes, and weighted adversarial learning is employed to selectively adapt domain distributions. In separated adversarial learning, the feature generator and the extended classifier are set against each other to construct more accurate hyperplanes between known and unknown fault modes. Comprehensive experimental results for three test rigs demonstrate that the proposed method achieves a promising performance and outperforms existing state-of-the-art open set domain adaptation methods.

##  Proposed Network 


![image](https://github.com/CHAOZHAO-1/Dual-adversarial-network-for-cross-domain-open-set-fault-diagnosis/blob/main/IMG/F1.png)

##  BibTex Citation


If you like our paper or code, please use the following BibTex:

```
@article{zhao2022dual,
  title={Dual adversarial network for cross-domain open set fault diagnosis},
  author={Zhao, Chao and Shen, Weiming},
  journal={Reliability Engineering \& System Safety},
  volume={221},
  pages={108358},
  year={2022},
  publisher={Elsevier}
}
```
