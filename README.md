# ThemeSpecificKG
## Introduction
The is the code for the KDD 2024 submit paper "Automated Construction of Theme-specific Knowledge Graphs". Given a theme (such as EV battery) and the theme-related raw documents, our tool can be used to generate a theme-specifc knowledge graph. We also provide two example themes which are EV battery and Hamas-attack on Israel.
## Preparation
1. Download a pre-generated wikipedia-word-frequency file to the path "src" from the [link](https://github.com/IlyaSemenov/wikipedia-word-frequency/blob/master/results/enwiki-2023-04-13.txt). You can also choose other language or version and set the filename in the PhraseMining.py.
2. (Optional) To apply [ZOE](https://github.com/CogComp/zoe/tree/master?tab=readme-ov-file), you should meet the requirement in ZOE and download the following [model.zip](https://cogcomp.seas.upenn.edu/Data/ccgPapersData/xzhou45/zoe/model.zip) (provided by ZOE) to the path src/bilm-tf. To make the framework run faster, you also can choose the model without the pre-trained ELMO in the settings of EntityTyping.py.
3. Run requirements `pip install -r requirements.txt`
4. set your openai token in the file RelationExtraction.py to run the GPT-4.
## Use the System
## test with my data
### Input
The input of the framework include (1) a theme (2) theme-related documents. I provide two datasets (1) electric vehicle battery in dataset/EVB and (2) Hamas-attack-Israel in dataset/HAI. 
### Inference and output
You can use run.sh to test with my data. The outputs of each step  with be saved into the dataset/xxx.json.
Since our framework constructs the KG step by step, we can also generate the outputs including 
(1) entity recognition: the offset of entity spans in the raw text.
(2) entity typing: the entity spans and their corresponding multiple categories
(3) relation extraction: the final results of triples
### using your own data
Preparation: 
(1) a theme, including the brief description of the theme
(2) Plain documents related to the theme. 
(3) (optional) predefined ontology. The format of ontology can follow the dataset/EVB/Battery_(electricity).csv
1. Save the file to dataset/[theme]/[theme].txt
2. (optional) Save the predefined ontology to dataset/[theme]/ontology.csv
3. Modify the settings of input in the src/PhraseMining.py
5. run the run.sh



