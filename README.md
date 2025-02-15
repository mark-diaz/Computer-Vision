#### Virtual environment instructions:

1. **Activate the Conda Virtual Environment:**
   - Open a terminal.
   - Use the command below to activate the cv conda environment:
     `conda activate cv`

2. **Run Your Script:**
   - Navigate to the directory containing this script:
     `cd /path/to/your/script`
   - Run the script using Python:
     `python your_script_name.py`

3. **Deactivate the Conda Virtual Environment:**
   - When you're done working in the environment, deactivate it by running:
     `conda deactivate`

4. **Keep Dependencies Isolated:**
   - When installing additional packages (like NumPy or others), always ensure the `cv` environment is active:
     `conda install <package_name>`
   - For example, to install matplotlib, you can run:
     `conda install matplotlib`

5. **Check Installed Packages:**
   - To see all packages installed in the environment:
     `conda list`

6. **Exit the Terminal:**
   - Once you're finished, you can exit the terminal by typing:
     `exit`

ASL Dataset: [Link](https://www.kaggle.com/datasets/grassknoted/asl-alphabet?resource=download-directory&select=asl_alphabet_train)


Some dependencies:
- numpy
- openCV
- matplotlib
- keras
- tensorflow
- pillow