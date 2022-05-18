from glob import glob
import cv2, skimage, os
import numpy as np
import scipy
from scipy.spatial import distance
import matplotlib.pyplot as plt
from tqdm import tqdm

NUM_MATCHES = 200

class OdometryClass:
    def __init__(self, frame_path):
        self.frame_path = frame_path
        self.frames = sorted(glob(os.path.join(self.frame_path, 'images', '*')))
        with open(os.path.join(frame_path, 'calib.txt'), 'r') as f:
            lines = f.readlines()
            self.focal_length = float(lines[0].strip().split()[-1])
            lines[1] = lines[1].strip().split()
            self.pp = (float(lines[1][1]), float(lines[1][2]))
            
        with open(os.path.join(self.frame_path, 'gt_sequence.txt'), 'r') as f:
            self.pose = [line.strip().split() for line in f.readlines()]

        self.detector=cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
        self.lk_params=dict(winSize  = (21,21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
    def imread(self, fname):
        """
        read image into np array from file
        """
        return cv2.imread(fname, 0)

    def imshow(self, img):
        """
        show image
        """
        skimage.io.imshow(img)

    def get_gt(self, frame_id):
        pose = self.pose[frame_id]
        x, y, z = float(pose[3]), float(pose[7]), float(pose[11])
        return np.array([[x], [y], [z]])

    def get_scale(self, frame_id):
        '''Provides scale estimation for mutliplying
        translation vectors
        
        Returns:
        float -- Scalar value allowing for scale estimation
        '''
        prev_coords = self.get_gt(frame_id - 1)
        curr_coords = self.get_gt(frame_id)
        return np.linalg.norm(curr_coords - prev_coords)

    def run(self):
        """
        Uses the video frame to predict the path taken by the camera
        
        The reurned path should be a numpy array of shape nx3
        n is the number of frames in the video  
        """
        path = []

        rotations = []
        transitions = []

        for i in range(len(self.frames[:10])):
            print(i)
            if i == 0:
                #pose = np.eye(4)
                #pose = np.zeros(shape=(3,4))
                pose = np.array(self.pose[0], dtype=np.float).reshape(3,4)
                pose = np.vstack([pose, [0, 0, 0, 1]])
                path.append(pose[:3, 3])
                rotations.append(np.eye(3))
                transitions.append(np.zeros((3,1)))
                print("initial pose: ", pose)
                print()
                print("initial point: ", path[-1])
                print()
                continue

            image1 = self.imread(self.frames[i - 1])
            image2 = self.imread(self.frames[i])
            p0 = self.detector.detect(image1)
            p0 = np.array([x.pt for x in p0], dtype=np.float32).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(image1, image2, p0, None, **self.lk_params)

            pts1 = p0[st == 1]
            pts2 = p1[st == 1]

            E, _ = cv2.findEssentialMat(pts2, pts1, self.focal_length, self.pp, cv2.RANSAC, 0.999, 1.0, None)
            _, R, t, _ = cv2.recoverPose(E, pts1, pts2, focal = self.focal_length, pp = self.pp)

            scale = self.get_scale(i)
            #if scale > .1:
                #t = t + scale * rotations[-1].dot(t)
                #R = R.dot(rotations[-1])
            
            rotations.append(R)
            transitions.append(t)

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.squeeze()

            pose = np.matmul(pose, np.linalg.inv(T))
            path.append(pose[:3, 3])

            print("Calculated pose: ", pose)
            print("Calculated point: ", path[-1])
            print("Real pose: ", np.array(self.pose[i]).reshape(3,4))
            print("Real point: ", self.get_gt(i).T)
            
        path = np.array(path)
        #print(path)
        #print(path.shape)
        return path
        
if __name__=="__main__":
    frame_path = 'video_train'
    odemotryc = OdometryClass(frame_path)
    path = odemotryc.run()
    #print(np.array(odemotryc.pose[-1]).reshape(3,4))
    #print(odemotryc.get_gt(-1))
    gt_path = np.array([odemotryc.get_gt(i).squeeze() for i in range(len(path))])
    
    ax = plt.axes(projection='3d')
    ax.plot3D(gt_path[:,0], gt_path[:,1], gt_path[:,2], color="green")
    ax.plot3D(path[:,0], path[:,1], path[:,2], color="red")
    plt.show()
    

