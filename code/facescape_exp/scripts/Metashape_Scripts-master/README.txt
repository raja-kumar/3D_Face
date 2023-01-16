You will need to install dlib and Metashape:
conda install -c conda-forge dlib
pip install Metashape-1.8.1-cp35.cp36.cp37.cp38-none-win_amd64.whl

The Metashape .whl file can be found at:
https://www.agisoft.com/downloads/installer/

If Metashape fails to install then run the following to find compatible tags:
pip debug --verbose
and rename the .whl file to "Metashape-1.8.0-<compatible tag>.whl

You will also need to download the shape predictor used by dlib and place it in
the root directory:
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Place the input image sets under input_images/ and then generate the meshes and
metashape project files:
python get_mesh.py

To get the landmarks, first fill out landmark_views.txt with a list of tuples
mapping image set numbers to camera view numbers of front-facing images so good
landmarks can be generated:
python get_landmarks.py
