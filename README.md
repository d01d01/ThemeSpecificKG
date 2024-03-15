# ThemeSpecificKG
## Introduction
The is the code for the KDD 2024 submit paper "Automated Construction of Theme-specific Knowledge Graphs". Given a theme (such as EV battery) and the theme-related raw documents, our tool can be used to generate a theme-specifc knowledge graph. We also provide two example themes which are EV battery and Hamas-attack on Israel.
## Preparation
1. Download a pre-generated wikipedia-word-frequency file to the path "code" from the [link](https://github.com/IlyaSemenov/wikipedia-word-frequency/blob/master/results/enwiki-2023-04-13.txt). You can also choose other language or version and set the filename in the PhraseMining.py.
2. (Optional) To apply [ZOE](https://github.com/CogComp/zoe/tree/master?tab=readme-ov-file), you should meet the requirement in ZOE and download the following [model.zip](https://cogcomp.seas.upenn.edu/Data/ccgPapersData/xzhou45/zoe/model.zip) (provided by ZOE) to the path code/bilm-tf. To make the framework run faster, you also can choose the model without the pre-trained ELMO in the settings of EntityTyping.py.
3. Run requirements `pip install -r requirements.txt`
## Use the System
### using your own data





