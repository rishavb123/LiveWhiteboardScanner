import cv2
from cv2utils.args import make_parser
from cv2utils.camera import make_camera_with_args


def prepare(frame):
    return frame


def preprocess(frames, raw):
    return frames[0]

if __name__ == '__main__':
    parser = make_parser()
    camera, args = make_camera_with_args(parser=parser, log=False, fps=15, res=(1280, 720))
    camera.stream(prepare=prepare, preprocess=preprocess, frames_stored=1)