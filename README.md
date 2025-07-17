# Assignment 4- Chapter 3.1-Statistics in Python 
### Scipy lecture Notes- Python Statistics- Walkthrough, Environment Generation/Updating, and Command line editing

 
## Project Description

This project was used to practice:

 - How to update git repository
 - How to create a reproducable environment.yml file
 - How to modify and update files from the command line using nano or vi
 - How to update an environment

Created a folder containing multiple practice CSV files and a detailed notebook working through the Scipy lecture notes for **Chapter 3.1-Statistics in Python** by Gael Varoquaux was created

 

**Table of Contents**

- Installation Instructions
- Usage
- Project Structure
- License
- Citations and Acknowledgements

 
## Installation Instructions

1. Create and clone your repository
- Begin by creating a Git Repository on GitHub
- Clone the repository to your local machine via the following command line "git clone https://github.com/your-username/your-repot-title.git

2. Create Files and Add the files to your repository
- Create .gitignore in the command line using a nano or vi command-- this file will provide a list of files/folders that Git should ignore. We will create a .gitignore with the following files to ignore.
  __pycache__/ 
*.pyc 
.ipynb_checkpoints/ 
.env/ 
.venv/ 
data/ 
*.csv 
*NOTE*: upload this file to your git repository

- Create a environment.yml file for conda environments only (as this was the only type in this project) with the following channels and dependencies named "stats-env"
name: stats-env
channels:
  - conda-forge
  - bioconda
  - defaults
dependencies:
  - python=3.10
  - jupyterlab
  - ipykernel
  - matplotlib
  - numpy
  - statsmodels
  - seaborn
  - scipy
  - pandas

3) Set up your environment with the following command lines to activate your .yml
cd Assignment4-Stats-Scripts (or whichever working directory you are in)
conda env create -f environment.yml
conda activate stats-env
jupyter lab

4) Create a folder labelled notebook and within create a stats_python.ipynb file

5) Create a detailed README.md (this file)

## Usage
How to use the code. Could include examples, expected inputs/outputs, or screenshots.

**How to use .gitignore**

  __pycache__/ #ignoring python cache files
*.pyc #ignoring python cahche files
.ipynb_checkpoints/ #ignoring notebook checkpoints
.env/ #ignoring environments
.venv/ #ignoring virtual environments
data/ #ignoring data files
*.csv #ignoring .csv files
Visualize your git ignore using nano .gitignore

**How to edit files from the command line using nano (which was the only method used in this project)**

To open a or create a file-- nano filename.fileabbreviation 
You will then be in an editable textframe. You can update your text or yml and then use the following commands to execute tasks for your file.
CTRL + O: Save the file ("Write Out")
Enter: Confirm the filename
CTRL + X: Exit nano
CTRL + K: Cut a line
CTRL + U: Paste a line
CTRL + W: Search

**Creating an environment, how to edit and make it usable**
1.Create an environment.yml file with the following channals for a conda environment
name: stats-env
channels:
  - conda-forge
  - bioconda
  - defaults
dependencies:
  - python=3.10
  - jupyterlab
  - ipykernel
  - matplotlib
  - numpy
  - statsmodels
  - seaborn
  - scipy
  - pandas
  
  You will then need to add this environment to your git repository and ensure you update it each time with any changes to your environment settings.
  Dependencies will vary based on project, these were the depndencies needed for this project and more can be added as needed. 
  *Note*: To activate your environment from the terminal you must first load miniconda which requires the following command for python 3.10: module load miniconda3/24.1.2-py310
- To activate your package: conda activate environment-name
- To install your package: conda install package-name 
- Export certain packages: conda env export --from-history > environment.yml

To set up your environment to use
1) Ensure you are in the correct working directory: cd your_working_directory e.g. Assignment4-Stats-Scripts
2) Create the environment using conda from your yml file: conda env create -f environment.yml
       *Note*: If the environment.yml is updated ensure you update your active environment with: conda env update -f environment.yml
3) Then activate the environment with: conda activate stats-env
4) Open jupyter lab with: juputer lab

### Basic functions for statistics in python using NumPy, SciPy, Seaborn, Pandas, statsmodels, matplotlib

*Note*: Here will be a list of useful functions-- **For detailed explanation of how to use these functions, the logic behind these functions, helpful figures and examples of their utility** see the **stats_python.ipynb** file in the Assignment4-Stats-Scripts notebook folder. Additional examples may be described in the notebook.

**a) Simple linear regression using scipy.stats.linregress**

from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt
#utilizes both numpy and matplotlib

#Sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 5, 7, 11])
#can also use data sets between paretheses, this is example data.
#Run regression
res = linregress(x, y)

print("Slope:", res.slope)
print("Intercept:", res.intercept)
print("R-squared:", res.rvalue**2)
print("p-value:", res.pvalue)
#the printing of the values gives the values calculated from the linregress function

#Plot
plt.scatter(x, y, label='Data')
plt.plot(x, res.intercept + res.slope * x, 'r', label='Fitted line')
plt.legend()
plt.show()

*Note* This was highly used in Assignment 2 and 3, please refer to these assignments to see how these functions were used.

**b) Creating a linear regression plots with seaborn**
To create a plot with multifunctionality and the ability to visualize subsets you should use the following functions and commands:

import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")
sns.lmplot(x="total_bill", y="tip", data=tips,
           hue="sex", robust=True,
           scatter_kws={"alpha":0.5})

plt.show()

*Note* with this you can subset the data by using the hue= function and creates a more complex regression analysis

If you are evaluating less complex data, which does not need to be delineated in to subsets you could use this lower level function of seaborn, see command here:

sns.regplot(x="total_bill", y="tip", data=tips, ci=95)
plt.show()

**c) Basic Statistical Functions using NumPy and SciPy**
scipy.stats.wilcoxon(arr1, arr2)- non-parametric test for paired samples
scipy.stats.ttest_ind(arr1, arr2)= two-sample t-test for comparing independent samples
scipy.stats.ttest_1samp(arr, value)= one-sample t-test to compare a mean to a known value
data.mean()- compute data mean
data. median()-compute data median
data. var()- compute variance
data. std()- compute standard deviation

**d) Pandas Functions**
pandas.read_csv( )- reads and imports a .CSV
df.groupby('category')['value'].mean()- calculates mean value per category
df.describe ()- summary statistics of all numeric columns, can specify certain columns based on categorical variables.
df.boxplot()- visualizes data using boxplot

**e) Matplotlib and Seaborn Visualization**
seaborn.pairplot(df, kind='reg', diag_kind='kde') - Pairwise scatterplots with regressions
plot.boxplot(data)- boxplot generation
plt.hist(data, bins=)- generates a histogram with modifiable sized bins.

## Project Structure

Assignment4-Stats-Scripts
    |_______ Notebooks
    |    |____ 3_1_1_1_SS.png
    |    |____ brain_size.csv
    |    |____ iris.csv
    |    |____ stats_python.ipynb
    |    |____ wages.txt
    |_______ environment.yml   
    |_______ README.md
    

    
## License 

This repository is intended for educational use only
 

## Acknowledgments and Citations
Based on exercises from: https://scipy-lectures.org/packages/statistics/index.html © The SciPy Lecture Notes authors. Lecture 3.1 by Gael Varoquaux

Lee Willerman, Robert Schultz, J. Neal Rutledge, Erin D. Bigler, In vivo brain size and intelligence, Intelligence, Volume 15, Issue 2, 1991, Pages 223-228, ISSN 0160-2896, https://doi.org/10.1016/0160-2896(91)90031-8.

Berndt, ER. The Practice of Econometrics. 1991. NY: Addison-Wesley.