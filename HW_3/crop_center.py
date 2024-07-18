import cv2

def center_crop(img, dim):
  """Returns center cropped image
  """
  width, height = img.shape[1], img.shape[0]
  
  crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
  crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
  
  middle_x, middle_y = int(width/2), int(height/2)
  cw2, ch2 = int(crop_width/2), int(crop_height/2) 
  crop_img = img[middle_y-ch2:middle_y+ch2, middle_x-cw2:middle_x+cw2]

  #cv2.imshow('returned_image',crop_img)
  #cv2.waitKey(0)
  #cv2.destroyAllWindows()

  return crop_img