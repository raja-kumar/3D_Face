import cv2
import os
from tqdm import tqdm

for size in (256, 512, 1024, 2048):
    print("size", size)
    for image_set in os.scandir("input_images"):
        print("image set", image_set.name)

        if not image_set.is_file():
            base_set_name = image_set.name[:image_set.name.index("_")]
            save_folder = os.path.join("input_images_resized", base_set_name + "_" + str(size))
    
            for image in tqdm(os.scandir(image_set.path)):
                if image.path.endswith(".jpg") and image.is_file():
                    img_obj = cv2.imread(image.path)
                    x, y, _ = img_obj.shape
        
                    if x > y:
                        rate = y / x
                        img_obj = cv2.resize(img_obj, (int(size*rate), size), interpolation=cv2.INTER_AREA)
                    else:
                        rate = x / y
                        img_obj = cv2.resize(img_obj, (size, int(size*rate)), interpolation=cv2.INTER_AREA)
        
                    img_obj = cv2.resize(img_obj, (y, x), interpolation=cv2.INTER_CUBIC)

                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    cv2.imwrite(os.path.join(save_folder, image.name), img_obj)