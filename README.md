# GeneLLM
GeneLLM is a large language model that seeks gene embeddings by utilizing human knowledge from different resources. As of now, the model only relies on text data available on the internet to generate initial embeddings that can be utilized for a variety of tasks such as Nuclear Localization, Conservation, and so on. 


## Installation
 Clone the repo and then run the following command to install all dependencies:
```
    cd GeneLLM
    conda env create --file requirements.yml
    conda activate DeepVul
```
Once the environment is set up correctly, you should be able to run the entire notebook without any problems. Please note that the code will be further refined at the time of submitting the paper such that people can run it from any framework/machine (i.e. from any command line).


## Description:
Below is a brief description of the  main directories in this GitHub repo: 
+ **GeneLLM.ipynb:** The main notebook that contains the fine-tuned models.
+ **genellm_scanpy.ipynb:** This notebook is for enrichment analysis.
+ **saved-figures:** The generated plots are saved to this directory.
+ **data:** This directory contains the data used by the model.
+ **DataExtraction:** The data extraction folder.
+ **requirements.yml:** This file is for setting up the working environment.




## License

MIT 

**Free Software**
