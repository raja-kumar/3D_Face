from getopt import getopt
import os
import sys

import Metashape
import PhotoScan

from select_img_dense2d import select_imgs

# export agisoft_LICENSE=/opt/agisoft/metashape-pro

# Directory constants
all_dir = 'C:/project/3d_face/Agisoft/Facescape_dense2d'
GENERATED_DIR = 'C:/project/3d_face/Agisoft/dense2d/'

AGGRO_FILTERING = Metashape.FilterMode.AggressiveFiltering  # type: ignore
OBJ = Metashape.ModelFormat.ModelFormatOBJ


def main():
    high_res = False

    opts, _ = getopt(sys.argv[1:], '-h', ['high-res'])

    if len(opts) > 0:
        for opt, _ in opts:
            if opt in ('-h', '--high-res'):
                high_res = True

    generateMeshes(high_res)



def generateMeshes(high_res):
    
    for res in os.listdir(all_dir):

        for person in os.listdir(os.path.join(all_dir, res)):

            c_names = select_imgs(person)

            for c_name in c_names:
                if c_name == 'all':
                    save_name = 'all'
                elif len(c_name) > 2:
                    save_name = str(len(c_name))
                elif len(c_name) == 2:
                    save_name = ''
                    for c in c_name:
                        save_name = save_name + '_' + str(c)

                # save_path = os.path.join(GENERATED_DIR, res, person + '_' + save_name + '.obj')
                save_path = os.path.join(GENERATED_DIR, res, person + '.obj')
                if os.path.exists(save_path):
                    print(save_path)
                    print('file exists.')

                else:
                    print(save_path)
                    if not os.path.exists(os.path.join(GENERATED_DIR, res)):
                        os.makedirs(os.path.join(GENERATED_DIR, res))
                    image_list = []
                    if c_name == 'all':
                        for image in os.listdir(os.path.join(all_dir, res, person)):
                            if image.split('.')[-1] == "jpg":
                                image_list.append(os.path.join(all_dir, res, person, image))
                    else:
                        for c in c_name:
                            image_list.append(os.path.join(all_dir, res, person, c+'.jpg'))

                    doc = Metashape.Document()
                    chunk = doc.addChunk()
                    chunk.addPhotos(image_list)

                    chunk.matchPhotos()

                    # calibration_path = 'C:/project/3d_face/Agisoft/cameras/'+ person + '.xml'
                    # chunk.importCameras(calibration_path)
                    chunk.alignCameras()
                    calibration_path = 'C:/project/3d_face/Agisoft/dense2d_cameras/'+ person + '.xml'
                    chunk.exportCameras(calibration_path)
                    
                    chunk.buildDepthMaps(downscale=16)

                    chunk.buildDenseCloud(point_colors=False, point_confidence=True)
                    chunk.exportPoints(save_colors=False, save_normals=False, binary=False, path=save_path)


if __name__ == '__main__':
    main()
