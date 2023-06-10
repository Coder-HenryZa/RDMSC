## Source code for the Information Sciences paper "Rumor Detection on Social Media through Mining the Social Circles with High Homogeneity"

### Requirements

Code developed and tested in Python 3.9 using PyTorch 1.10.2 and Torch-geometric 2.2.0. Please refer to their official websites for installation and setup.

Some major dependencies are as follows:

```
certifi==2023.5.7
charset-normalizer==3.1.0
colorama==0.4.6
contourpy==1.0.7
cycler==0.11.0
emoji==2.2.0
fonttools==4.39.4
idna==3.4
importlib-resources==5.12.0
joblib==1.2.0
kiwisolver==1.4.4
MarkupSafe==2.1.2
matplotlib==3.7.1
numpy==1.24.3
packaging==23.1
pandas==2.0.1
Pillow==9.5.0
psutil==5.9.5
PyMySQL==1.0.3
pyparsing==3.0.9
python-dateutil==2.8.2
pytz==2023.3
requests==2.30.0
scikit-learn==1.2.2
scipy==1.10.1
six==1.16.0
threadpoolctl==3.1.0
tqdm==4.65.0
typing_extensions==4.5.0
tzdata==2023.3
urllib3==2.0.2
zipp==3.15.0
```

### Datasets

Data of Twitter15 and Twitter16 social interaction graphs follows this paper:

Tian Bian, Xi Xiao, Tingyang Xu, Peilin Zhao, Wenbing Huang, Yu Rong, Junzhou Huang. Rumor Detectionon Social Media with Bi-Directional Graph Convolutional Networks. AAAI 2020.

The raw datasets can be respectively downloaded from https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0.

User information was crawled by the Twitter developer tool Tweepy around February 2022, when restrictions were not particularly strict. twitter15_User_Information.csv, Twitter15_User_Friends.csv and Twitter15_Ego_ Relationships.csv represent user information, friend information, and connections between them, respectively. The meaning of each field in the tables can be found as follows:

https://developer.twitter.com/en/docs/twitter-api/v1/data-dictionary/object-model/user.

In order to meet Twitter's privacy protocol, we have removed some fields and limited the number of friends per user.

### Run

```
# Data pre-processing
python ./utils/getInteractionGraph.py Twitter15
python ./utils/getInteractionGraph.py Twitter16
python ./utils/getEgoGraph.py Twitter15
python ./utils/getEgoGraph.py Twitter16
# run
python RDMSC_Run.py
```

### Citation

If you find this repository useful, please kindly consider citing the following paper:

```
@article{ZHENG2023119083,
title = {Rumor detection on social media through mining the social circles with high homogeneity},
journal = {Information Sciences},
volume = {642},
pages = {119083},
year = {2023},
issn = {0020-0255},
doi = {https://doi.org/10.1016/j.ins.2023.119083},
url = {https://www.sciencedirect.com/science/article/pii/S0020025523006680},
author = {Peng Zheng and Zhen Huang and Yong Dou and Yeqing Yan},
keywords = {Rumor detection, Social media, Social circles, Homogeneity}
}
```

