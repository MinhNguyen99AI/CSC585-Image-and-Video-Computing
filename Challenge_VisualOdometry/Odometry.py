from glob import glob
import cv2, skimage




class OdometryClass:
    def __init__(self, frame_path):
        self.frame_path = frame_path
        self.frames = sorted(glob(self.frame_path+'/*'))
        
    def imread(self, fname):
        """
        read image into np array from file
        """
        return cv2.imread(fname)

    def imread_bw(self, fname):
        """
        read image as gray scale format
        """
        return cv2.cvtColor(self.imread(fname), cv2.COLOR_BGR2GRAY)

    def imshow(self, img):
        """
        show image
        """
        skimage.io.imshow(img)

    
    def run(self):
        """
        Uses the video frame to predict the path taken by the camera

        The reurned path should be a numpy array of shape nx3
        n is the number of frames in the video  
        """

        raise NotImplementedError
        

if __name__=="__main__":
    frame_path = 'undistorted_images'
    odemotryc = OdometryClass(frame_path)
    path = odemotryc.run()
    print(path,path.shape)
