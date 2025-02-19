import cv2
import pytz
import traceback
import numpy as np
from datetime import datetime
import jspl_hsm_coil_id_mapping.constants
from ripikvisionpy.commons.kinter.Utils import Utils
import logging

utils = Utils()
DB = None
DB_UTILS_OS = 'UBUNTU'

def get_camera_data(client_meta, material, camera_id):
    try:
        camera_info = client_meta['toolsV2'][material]
        for row in camera_info:
            if row['cameraId'] == camera_id:
                return row
        return None

    except Exception as e:
        logging.error("Exception in getting camera data : " + str(e))
        return None


def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """
    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )
    return result

def get_utc_timestamp(milli=False):
    """
    :param milli: If resp is needed in ms epoch
    :return utc epoch:
    """ 
    if milli:
        utc_timestamp = int(datetime.now(pytz.utc).timestamp()*1000)
    else:
        utc_timestamp = int(datetime.now(pytz.utc).timestamp())
    return utc_timestamp

def uploadImgS3(img, s3, bucket, key):
    data = cv2.imencode('.jpg', img)[1].tostring()
    s3.put_object(Body = data, Bucket = bucket, Key = key)
    url = f"https://{bucket}.s3.ap-south-1.amazonaws.com/{key}"
    return url

def get_response_with_s3_links(
    s3,
    record,
    aws_bucket_name,
):
    created_at =int(get_utc_timestamp(milli=True))
    timestamp_epoch = ((created_at / 1000) + 19800)
    timestamp_obj = datetime.utcfromtimestamp(timestamp_epoch)

    timestamp = timestamp_obj.strftime('%Y-%m-%d %H:%M:%S')
    date = str(timestamp_obj.date())
    hour = str(timestamp_obj.hour)
    time_ = str(timestamp_obj.time().strftime('%H-%M-%S'))
    
    record['createdAt'] = created_at
    record['timestamp'] = timestamp
    if record['isAlert']:
        subfolder = 'alert'
    else:
        subfolder = 'no'
    image_tags = ['originalImage', 'annotatedImage']
    resize_prop = 100 / 100
    for tag in image_tags:
        if tag in record:
            if not isinstance(record[tag], str):
                record[tag] = cv2.resize(record[tag], (0, 0), None, resize_prop, resize_prop)
                object_key = f'{record["clientId"]}/loc1/{record["usecase"]}/{record["cameraGrpId"]}/{record["cameraId"]}/{subfolder}/{tag}/{date}/{hour}/{record["cameraId"]}_{tag}_{date}_{time_}.jpg'
                signedUrl = uploadImgS3(img = record[tag], s3 = s3, bucket = aws_bucket_name, key = object_key)
                record[tag] = signedUrl
                print(f'S3 Upload Complete: {tag}')
    return record

def push_data_to_mongo(response: dict, client_meta_wrapper, client_meta):
    try:
        global DB
        mongodb_info = client_meta_wrapper.extract_mongodb_info(client_meta)
        if DB is None:
            print("Connecting to DB: " + str(mongodb_info['dbName']))
            DB = utils.get_mongodb_instance(mongodb_info['dbUrl'], mongodb_info['dbName'], DB_UTILS_OS)
        DB[mongodb_info['coll']['history']].insert_one(response)
        print('INFO: Data pushed to MongoDB!')
    except Exception as e:
        traceback.format_exc()
        print(f'ERROR: Failed to push data to Mongo!')