# Image-Segementation
Image Segmentation using Kmeans and DBSCAN algorithm



# Image Segmentation

Image segmentation involves converting an image into a collection of regions of pixels that are represented by a mask or a labeled image.

## Features

- Reads the image from user.
- Performs image segmentation using algorithms : Kmeans and DBScan

## Tech
Script uses following tech stack:

- [Python] - An interpreted high-level general-purpose programming language.

## Installation

Clustering Algorithm requires [Python](https://www.python.org) 3+ to run.

- Scripts used : main.py, algo.py, log.py
- Place the scripts into a directory of choice.
- Navigate to the directory and run the following:

```
pip install -r requirements.txt

```
or 

```
pip install the modules mentioned in the requirements.txt file

```
## Usage

- Navigate to the folder where the script is placed.
- Run main.py script.
- A pop up window will appear where the user can select the image to be segmented.
- The Class "Kmeans()" in "algo.py"  performs the Kmeans algorithm for image segmentation.  takes in It 2 arguments as inputs.
- The function "dbscan()"  in "algo.py" performs the DBscan algorithm. It takes 3 arguments as input.

- Logs are generated in the working folder with name "clustering_main_log".
- Log file contains the information about the Filename used, Algorithm used ( Kmeans or DBscan )
- K value used by the Kmeans algorithm, parameters : epsilon and minpts of DBscan, start and end time of the corresponding algorithm used.




