from getopt import getopt
import os
import sys

import Metashape

# export agisoft_LICENSE=/opt/agisoft/metashape-pro
# & 'C:\Program Files\Agisoft\Metashape Pro\metashape.exe' -r .\get_mesh.py

# Directory constants
IMAGES_DIR = "input_images"
GENERATED_DIR = "output_meshes"

AGGRO_FILTERING = Metashape.FilterMode.AggressiveFiltering  # type: ignore
OBJ = Metashape.ModelFormat.ModelFormatOBJ  # type: ignore


def main():
    high_res = False
    save_cameras = False
    view = 60

    opts, _ = getopt(sys.argv[1:], '-h-v:-s', ['high-res', 'views', 'save-cameras'])

    if len(opts) > 0:
        for opt, arg in opts:
            if opt in ('-h', '--high-res'):
                high_res = True
            if opt in ('-s', '--save-cameras'):
                save_cameras = True
            if opt in ('-v', '--views'):
                view = int(arg)

    generateMeshes(high_res, save_cameras, view)

def select_imgs(image_set, view=2):
    c_names = [[]]
    if view == 2:
        # # 2-view cases, 20*3
        if image_set == '1':
            c_names = [['1', '52'], ['1', '43'], ['52', '43']]
        elif image_set == '2':
            c_names = [['16', '17'], ['16', '21'], ['21', '17']]
        elif image_set == '3':
            c_names = [['16', '17'], ['16', '21'], ['21', '17']]
        elif image_set == '4':
            c_names = [['1', '43'], ['1', '52'], ['43', '52']]
        elif image_set == '5':
            c_names = [['16', '20'], ['16', '15'], ['15', '20']]
        elif image_set == '6':
            c_names = [['17', '18'], ['17', '20'], ['18', '20']]
        elif image_set == '7':
            c_names = [['16', '17'], ['16', '19'], ['17', '19']]
        elif image_set == '9':
            c_names = [['17', '19'], ['17', '21'], ['19', '21']]
        elif image_set == '11':
            c_names = [['23', '24'], ['23', '25'], ['24', '25']]
        elif image_set == '13':
            c_names = [['49', '51'], ['49', '54'], ['51', '54']]
        elif image_set == '14':
            c_names = [['58', '60'], ['58', '67'], ['60', '67']]
        elif image_set == '15':
            c_names = [['58', '60'], ['58', '67'], ['60', '67']]
        elif image_set == '16':
            c_names = [['24', '33'], ['24', '34'], ['33', '34']]
        elif image_set == '17':
            c_names = [['16', '17'], ['16', '21'], ['17', '21']]
        elif image_set == '18':
            c_names = [['23', '25'], ['23', '34'], ['25', '34']]
        elif image_set == '19':
            c_names = [['1', '10'], ['1', '43'], ['10', '43']]
        elif image_set == '20':
            c_names = [['56', '58'], ['56', '65'], ['58', '65']]
        elif image_set == '121':
            c_names = [['44', '51'], ['44', '54'], ['51', '54']]
        elif image_set == '122':
            c_names = [['40', '44'], ['40', '48'], ['44', '48']]
        elif image_set == '212':
            c_names = [['47', '51'], ['47', '55'], ['51', '55']]
    elif view == 3:
        # # 3-view cases, 20*1
        if image_set == '1':
            c_names = [['1', '4', '8']]
        elif image_set == '2':
            c_names = [['10', '16', '21']]
        elif image_set == '3':
            c_names = [['7', '17', '21']]
        elif image_set == '4':
            c_names = [['1', '4', '8']]
        elif image_set == '5':
            c_names = [['6', '16', '20']]
        elif image_set == '6':
            c_names = [['7', '17', '20']]
        elif image_set == '7':
            c_names = [['7', '17', '21']]
        elif image_set == '9':
            c_names = [['7', '17', '21']]
        elif image_set == '11':
            c_names = [['24', '25', '32']]
        elif image_set == '13':
            c_names = [['46', '47', '51']]
        elif image_set == '14':
            c_names = [['27', '33', '34']]
        elif image_set == '15':
            c_names = [['24', '25', '32']]
        elif image_set == '16':
            c_names = [['1', '4', '8']]
        elif image_set == '17':
            c_names = [['10', '17', '21']]
        elif image_set == '18':
            c_names = [['1', '7', '8']]
        elif image_set == '19':
            c_names = [['7', '43', '56']]
        elif image_set == '20':
            c_names = [['57', '58', '61']]
        elif image_set == '121':
            c_names = [['44', '49', '50']]
        elif image_set == '122':
            c_names = [['40', '44', '45']]
        elif image_set == '212':
            c_names = [['47', '51', '52']]

    return c_names

def generateMeshes(high_res, save_cameras, view):
    # iterate over every face's image set
    for image_set in os.scandir(IMAGES_DIR):
        if not image_set.is_file():
            base_set_name = image_set.name[:image_set.name.index("_")]
            image_lists = []

            if view != 60:
                image_lists = select_imgs(base_set_name, view)

                for i in range(len(image_lists)):
                    for j in range(view):
                        image_lists[i][j] = os.path.join(image_set.path, image_lists[i][j] + ".jpg")
            else:
                image_list = []

                # create a list of all the image paths
                for image in os.scandir(image_set.path):
                    if image.path.endswith(".jpg") and image.is_file():
                        image_list.append(image.path)

                image_lists = [image_list]

            for image_list in image_lists:
                doc = Metashape.Document()  # type: ignore
                # create a chunk for the document and add the photos for this set
                chunk = doc.addChunk()
                chunk.addPhotos(image_list)
                calibration_path = os.path.join("calibration", base_set_name + ".xml")

                matchFailed = False
                try:
                    if high_res:
                        # align photos using highest accuracy
                        chunk.matchPhotos(downscale=0)
                    else:
                        chunk.matchPhotos()
                except RuntimeError as e:
                    matchFailed = True
                    print(e)

                if not matchFailed:
                    # load calibration for all cameras if available
                    if os.path.exists(calibration_path):
                        chunk.importCameras(calibration_path)

                    chunk.alignCameras()

                    if high_res:
                        # build dense cloud using ultra quality, aggressive filtering
                        chunk.buildDepthMaps(downscale=1, filter_mode=AGGRO_FILTERING)
                    else:
                        chunk.buildDepthMaps()

                    buildCloudFailed = False
                    try:
                        chunk.buildDenseCloud()
                    except Exception as e:
                        buildCloudFailed = True
                        print(e)

                    if not buildCloudFailed:
                        # build mesh
                        chunk.buildModel()

                        set_name = image_set.name

                        if view != 60:
                            for image_path in image_list:
                                slash_i = image_path.rfind(os.path.sep)
                                set_name += "_" + image_path[slash_i + 1:image_path.rindex(".")]

                        # save the mesh as a ply file without any colors
                        chunk.exportModel(save_colors=False, format=OBJ, binary=False,
                                        path=os.path.join(GENERATED_DIR, "obj",
                                                            set_name + '.obj'))

                        # save calibration for all cameras
                        if save_cameras:
                            chunk.exportCameras(calibration_path)

                        # save document for future reference as an archive project file
                        doc.save(os.path.join(GENERATED_DIR, "project",
                                set_name + ".psz"))


if __name__ == '__main__':
    main()