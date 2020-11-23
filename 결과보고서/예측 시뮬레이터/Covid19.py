class MaskAreaDetector():

    def __init__(self):

        import cv2
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontface.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        self.glass_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
        self.eyesplit_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
        return

    def advanced_eye_detect(self,img,cascade=None,info='normal'):
        import cv2
        
        if info=='normal':
            cascade = self.eye_cascade
        
        ret_val=[]
        
        img_gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(img_gray)
        
        if len(faces)!=1:
            return ret_val
        
        for (x,y,w,h) in faces:
            roi_gray = img_gray[y:y+h, x:x+w]
            eyes =cascade.detectMultiScale(roi_gray,1.1,4)
        
        if not (len(eyes)==2):    
            if info == 'normal':
                return self.advanced_eye_detect(img,self.eyesplit_cascade,'split')
            elif info == 'split':
                return self.advanced_eye_detect(img,self.glass_cascade,'glasses')
            elif info == 'glasses':
                return ret_val
            
        ret_val = eyes
        return ret_val 

    def get_rotated_image(self,img,eyes):
        import cv2
        import numpy as np
        
        eye_1 , eye_2 = eyes

        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2
            right_eye = eye_1
            
        left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
        right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
        
        left_eye_x , left_eye_y = left_eye_center 
        right_eye_x , right_eye_y = right_eye_center

        
        delta_x = right_eye_x - left_eye_x
        delta_y = right_eye_y - left_eye_y
        
        if not delta_x or delta_y:
            return img
        
        angle=np.arctan(delta_y/delta_x)
        angle = (angle * 180) / np.pi
        
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, (angle), 1.0)
        rotated_img = cv2.warpAffine(img, M, (w, h))
      
        return rotated_img

    def calculate_rotated_eyes(self,rotated_img,cascade=None,info='normal'):
        import cv2
        import numpy as np

        if info=='normal':
            cascade = self.eye_cascade

        ret_val=[]
        
        rotated_gray = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(rotated_gray, 1.1, 4)
        
        if len(faces)!=1:
            return ret_val
        
        for (x,y,w,h) in faces:

            black=np.zeros(rotated_gray.shape,dtype='uint8')
            black[y:y+h, x:x+w]=rotated_gray[y:y+h, x:x+w]

            rotated_eyes =cascade.detectMultiScale(black,1.1,4)


        if not (len(rotated_eyes)==2):    
            if info == 'normal':
                return self.calculate_rotated_eyes(rotated_img,self.eyesplit_cascade,'split')
            elif info == 'split':
                return self.calculate_rotated_eyes(rotated_img,self.glass_cascade,'glasses')
            elif info == 'glasses':
                return ret_val
        
        ret_val = rotated_eyes
        
        return ret_val

    def extract_facial_mask_area(self,rotated_img,rotated_eyes):
        import cv2
        import numpy as np
        
        eye_1 , eye_2 = rotated_eyes

        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2
            right_eye = eye_1
        
        left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
        right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
        
        left_eye_x , left_eye_y = left_eye_center 
        right_eye_x , right_eye_y = right_eye_center
        
        delta_x = right_eye_x - left_eye_x
        delta_y = right_eye_y - left_eye_y
        
        L = np.sqrt(delta_x**2 + delta_y**2)
        xpad_L , xpad_R = int(0.6*L) , int(1.6*L)
        ypad_U , ypad_D = int(0.6*L) , int(1.8*L)
        
        ROI = rotated_img[left_eye_y-ypad_U:left_eye_y+ypad_D,left_eye_x-xpad_L:left_eye_x+xpad_R]
        ROI_resized = cv2.resize(ROI,(120,140))
        mask_area = ROI_resized[50:140,0:120]
        
        return mask_area
