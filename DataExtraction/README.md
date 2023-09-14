To join in the extraction of the data required for GeneLLM, follow the following steps:


Install Chromewebdriver: 

To do this,

1. Ensure you have chrome browser installed
2. then open your chrome browser, go to the settings, and click "About"
or paste this  "chrome://settings/help" in your search bar

3. Check the version of your chrome,
4. Then visit "https://chromedriver.chromium.org/downloads" to download the chromedriver that matches your chrome browser version and Laptop operating system.

5. If the downloaded driver needs extraction, Extract it:

     Then copy the driver into the the extraction folder that contains other file.


* Open your extraction folder in VSCODE, or an editor that can view jupyter notebook.

\
Description of each file in the folder:
* The "extract.ipynb" file is the notebook where the extraction codes are
* The "genes.csv" file contains list of all genes you are to extract summary data for.
* The  "input.csv" file: This file is created in order for you to extract in batches. You can copy all the genes present in the genes.csv file at once, or copy it in batches. # The first 30 genes has already be copied into this file for you for testing. 
* Make sure you save using "ctrl S" each time you make any changes in a file.
* The "output.tsv" is where the output of the extraction will be stored after running your code.

\
To run the code,
* Ensure the list of gene you want to extract its summary is present in the "input.csv" file, and saved.
* Ensure your chromedriver is also present in your folder.
* Open the extract.ipynb file
* The first cell contains the code to install all needed libraries. You can install them from terminal or anywhere thats convinient for you.
* The extraction  codes are in the second and third cell:
1. second cell for macbook users
2. and third cell for linux and windows users.


\
Run the code in the cell, and wait untill its done to view the output file.



