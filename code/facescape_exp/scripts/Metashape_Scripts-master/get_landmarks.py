import os
import numpy

import dlib
import Metashape

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Face predictor download
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2


def main():
    set_to_views = open('landmark_views.txt', 'r').readlines()

    for set_and_view in set_to_views:
        image_set, view_num = set_and_view.strip().split()

        # get the specified camera
        doc = Metashape.Document()  # type: ignore
        project_dir = os.path.join(
            "output_meshes", "project", image_set + ".psz")
        doc.open(project_dir)
        chunk = doc.chunks[0]
        camera = chunk.cameras[int(view_num)]

        landmarks_2D = get_2D_landmarks(camera.photo.path)
        landmarks_3D = get_3D_landmarks(chunk, camera, landmarks_2D)

        numpy.save(os.path.join('landmarks', image_set + '.npy'), landmarks_3D)
        # save the markers to the project
        doc.save(project_dir)


def get_2D_landmarks(image_path):
    # TODO move up into main if necessary
    detector = dlib.get_frontal_face_detector()  # type: ignore
    predictor = dlib.shape_predictor(PREDICTOR_PATH)  # type: ignore
    img = dlib.load_rgb_image(image_path)  # type: ignore
    dets = detector(img, 1)
    shape = predictor(img, dets[0])

    landmarks = []

    # get only the 51 landmarks needed for FLAME
    for i in range(17, 68):
        part = shape.part(i)
        landmarks.append((part.x, part.y))

    return landmarks


def get_3D_landmarks(chunk, camera, landmarks_2D):
    # clear all markers
    chunk.remove(chunk.markers)
    landmarks_3D = numpy.empty([51, 3])

    for i, landmark in enumerate(landmarks_2D):
        imgX = landmark[0]
        imgY = landmark[1]

        sensor = camera.sensor
        point2D = Metashape.Vector([imgX, imgY])  # type: ignore
        point3D = chunk.model.pickPoint(
            camera.center, camera.transform.mulp(
                sensor.calibration.unproject(point2D)))
        chunk.addMarker(point=point3D)

        landmarks_3D[i] = [point3D[0], point3D[1], point3D[2]]

    return landmarks_3D


if __name__ == '__main__':
    main()
