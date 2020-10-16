This project is a data exploratory work on Atom Probe Tomography scan of microstructures. The project acknowledges and adapts the data importing tools from
https://github.com/oscarbranson/apt-tools

##### How to use:

Step 1: Download the whole folder into a folder of choice. Create a subfolder named 'Data' inside the main folder, and download the demo datafile into the new subfolder using this [link](https://buffalo.box.com/s/iy8my7kzpyplcty67xn9pixf08mm1dj4). 

Step 2: Open terminal and use conda to create an environment as:

`conda create -n atomprobe python=3.7
 conda activate atomprobe` 

Step 3: Navigate to the folder where the codes are copied and type:

`pip install requirements.txt`

Step 4. Once all the libraries are successfully installed, run the demo using the following code:

`python app.py`

Please use the following article to cite our work:

Mukherjee, Arpan, Scott Broderick, and Krishna Rajan. "Modularity optimization for enhancing edge detection in microstructural features using 3D atomic chemical scale imaging." Journal of Vacuum Science & Technology A: Vacuum, Surfaces, and Films 38.3 (2020): 033207.
