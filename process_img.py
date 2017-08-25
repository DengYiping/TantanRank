from PIL import Image
import os
import face_recognition
import json

RAW_IMG_DIC = './images/'
CROPPED_IMG_DIC = './cropped_img/'

def crop_face(raw_img_file):
    """Find all the faces and crop the file in images folder to 128 * 128
    """
    print('---------------------------')
    print('Processing image {}'.format(raw_img_file))
    raw_img = face_recognition.load_image_file(RAW_IMG_DIC + raw_img_file)
    face_locations = face_recognition.face_locations(raw_img)
    print('{} faces found in this image'.format(len(face_locations)))
    count = 0
    for face_loc in face_locations:
        top, right, bottom, left = face_loc
        img_save_name = '{}_{}'.format(str(count), raw_img_file)
        face_img = raw_img[top:bottom, left:right]
        pil_image = Image.fromarray(face_img)
        resized_img = pil_image.resize((128,128))
        resized_img.save(CROPPED_IMG_DIC + img_save_name)
def get_rank(fname):
    path = CROPPED_IMG_DIC + fname
    Image.open(path).show()
    tag = int(input('Enter rank:'))
    return tag

def crop_all():
    raw_img_list = os.listdir(RAW_IMG_DIC)
    for f_name in raw_img_list:
        if '.png' in f_name:
            crop_face(f_name)
def rank_all():
    cropped_imgs = os.listdir(CROPPED_IMG_DIC)
    tags = {}
    for img in cropped_imgs:
        if '.png' in img:
            tags[img] = get_rank(img)
    with open('tags.json', 'w') as tag_file:
        tag_json = json.dumps(tags)
        tag_file.write(tag_json)


if __name__ == '__main__':
    crop_all()
    rank_all()
