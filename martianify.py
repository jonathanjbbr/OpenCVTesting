import cv2

def martianify(original_image_path):
    # Load the face detection cascade file.
    face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml')
    # Load our Martian as foreground image with alpha transparency.
    # The -1 reads the alpha transparency of our image otherwise
    # known as the face hole.
    foreground_image = cv2.imread('imgmask/martianmask.webp', -1)
    # Create foreground mask from alpha transparency.
    foreground_mask = foreground_image[:, :, 3]
    cv2.imwrite('tmp/m01-foreground_mask.jpg',foreground_mask)
    # Create inverted background mask.
    background_mask = cv2.bitwise_not(foreground_mask)
    cv2.imwrite('tmp/m02-background_mask.jpg',background_mask)
    # Convert foreground image to BGR.
    foreground_image = foreground_image[:, :, 0:3]
    cv2.imwrite('tmp/m03-foreground_image.jpg',foreground_image)
    # Declare foreground size.
    foreground_size = 600
    foreground_ratio = float(foreground_size)
    # Declare background size and padding.
    background_size = 1100
    padding_top = ((background_size - foreground_size) / 3) * 2
    padding_bottom = background_size - padding_top
    padding_left = (background_size - foreground_size) / 2
    padding_right = (background_size - foreground_size) / 2
    # Capture selfie image in OpenCV.
    cv_image = cv2.imread(original_image_path)
    # Find that face.
    faces = face_cascade.detectMultiScale(
        cv_image,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
        )
    # Iterate over each face found - roi: region of interest
    i=10
    for (x1, y1, w, h) in faces:
        # Extract image of face.
        x2 = x1 + w
        y2 = y1 + h
        face_roi = cv_image[y1:y2, x1:x2]
        cv2.imwrite('tmp/m04.'+ str(i) +'+1-face_roi.jpg',face_roi)
        # Resize image of face.
        ratio = foreground_ratio / face_roi.shape[1]
        dimension = (foreground_size, int(face_roi.shape[1] * ratio))
        face = cv2.resize(face_roi, dimension, interpolation=cv2.INTER_AREA)
        cv2.imwrite('tmp/m04.'+ str(i) +'+2-face.jpg',face)
        # Add padding to background image
        background_image = cv2.copyMakeBorder(face,
                                              padding_top,
                                              padding_bottom,
                                              padding_left,
                                              padding_right,
                                              cv2.BORDER_CONSTANT)
        cv2.imwrite('tmp/m04.'+ str(i) +'+3-background_image.jpg',background_image)
        # Region of interest for Martian from background proportional to martian size.
        background_src = background_image[0:background_size, 0:background_size, :3]
        cv2.imwrite('tmp/m04.'+ str(i) +'+4-background_src.jpg',background_src)
        # roi_bg contains the original image only where the martian is not
        # in the region that is the size of the Martian.
        dimension2=(background_size,background_size)
        background_mask2 = cv2.resize(background_mask, dimension2, interpolation=cv2.INTER_AREA)
        cv2.imwrite('tmp/m04.'+ str(i) +'+5-background_mask2.jpg',background_mask2)
        roi_bg = cv2.bitwise_and(background_src, background_src, mask=background_mask2)
        cv2.imwrite('tmp/m04.'+ str(i) +'+6-roi_bg.jpg',roi_bg)
        # roi_fg contains the image of the Martian only where the Martian is
        roi_fg = cv2.bitwise_and(foreground_image, foreground_image, mask=foreground_mask)
        cv2.imwrite('tmp/m04.'+ str(i) +'+7-roi_fg.jpg',roi_fg)
        roi_bg = cv2.resize(roi_bg, (foreground_size ,foreground_size ), interpolation=cv2.INTER_AREA)
        cv2.imwrite('tmp/m04.'+ str(i) +'+8-roi_bg.jpg',roi_bg)
        # Join the roi_bg and roi_fg.
        #print roi_bg.shape, roi_fg.shape
        dst = cv2.add(roi_bg, roi_fg) 
        cv2.imwrite('tmp/m05.final-'+ str(i) +'+9-dst.jpg',dst)
        cv2.imshow("Look mom, I'm the Martian!", dst)
        i=i+1
        cv2.waitKey()


martianify('imgdata/fam.jpg')

