from data import Images
import os

data_path = os.path.join(os.path.dirname(__file__), 'data')
with Images(os.path.join(data_path, 'images.tar')) as images:
    path = images.paths[3]
    image = images[path]
    print("read image {} of shape {}".format(path, image.shape))
# read image "person_0013/channel_0081/seq


from data import read_pose
#paths, keypoints, scores = read_pose('pose.pkl')
paths, keypoints, scores = read_pose(os.path.join(data_path, 'pose.pkl'))
print(paths.__len__())
print(keypoints.shape)
print(scores.shape)


from data import read_signatures
pathsSig, signatures = read_signatures(os.path.join(data_path, 'signatures.pkl'))
print(pathsSig.__len__())
print(signatures.shape)




