## About

Here I have tried to make a pipeline to get dense 2d landmarks using 2 and 3 input images.

## input format
- see ./input/<test_set_no>_<no_of_image>
- each folder contains input RGB images, agisoft generated scan & camera calibaration (in xml folder), triangulated scan using meshlab, flame 3D, and projected back flame 3D (to align with input scan)

## output format
- see ./output/<test_set_no>_<no_of_image>
- each folder contains the detected 2D lmk points (projected from 3D) and the points projected on image

## Steps to run end to end setup
- pick images (preferrably from same camera. or else feature point matching may fail in agisoft)
- use agisoft feature point mapping to find the dense cloud and camera parameters (in metashape software do -> align photos -> build dense cloud)
- use meshlab to convert the dense cloud to a triangualted mesh (open scan in meshlab and do the below steps)
    - filters -> normals, curvatures and orientation -> calculate normals for point set
    - filters -> remeshing, simplication and reconstruction -> surface reconstion: ball pivoting
    - export the mesh and save
- find the 51 point landmark as per convention given here (https://github.com/TimoBolkart/TF_FLAME#landmarks). there are couple of ways to do this 
    - do manually using pickpoint feature in meshlab 
    - use standard 68 lmk detection and find the 3D landmark using camera parameters. (check scripts folder)
- do the flame fitting (https://colab.research.google.com/drive/1mKvWGVHEl4KvQrwRmGogcITY0TQeievQ#scrollTo=bfuO2VV96nLW)
- project back the flame 3D to input scan (run ! python align_back.py in the above notebook)
- get the 2D projection using the camera calibration from step2 (run read.py)
