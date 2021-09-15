import numpy as np
import cv2
from cv2utils.args import make_parser
from cv2utils.camera import make_camera_with_args

# hard coded calibration
u1r = 18.5 / 31
v1r = 1.75 / 17
u2r = 17.4 / 31
v2r = 7.5 / 17
u3r = 28 / 31
v3r = 0.5 / 17
u4r = 25 / 31
v4r = 10 / 17

output_res = (300, 450, 3)
h, w, _ = output_res

def prepare(frame):
    # return frame
    input_h, input_w, _ = frame.shape
    u1, u2, u3, u4 = int(u1r * input_w), int(u2r * input_w), int(u3r * input_w), int(u4r * input_w)
    v1, v2, v3, v4 = int(v1r * input_h), int(v2r * input_h), int(v3r * input_h), int(v4r * input_h)

    def transform(x, y):
        u_1 = x / w * (u1 - u2) + u2
        u_2 = x / w * (u3 - u4) + u4
        v_1 = y / h * (v1 - v2) + v2
        v_2 = y / h * (v3 - v4) + v4
        u_3 = x / w * (u3 - u1) + u1
        u_4 = x / w * (u4 - u2) + u2
        v_3 = y / h * (v3 - v1) + v1
        v_4 = y / h * (v4 - v2) + v2

        m_12 = (v_2 - v_1) / (u_2 - u_1)
        m_34 = (v_4 - v_3) / (u_4 - u_3)

        u = (v_3 - v_1 + m_12 * u_1 - m_34 * u_1) / (m_12 - m_34)
        v = m_12 * (u - u_1) + v_1

        u = int(u)
        v = int(v)
        return u, v

    output = np.zeros(output_res)

    for x in range(w):
        for y in range(h):
            u, v = transform(x, y)
            output[y][x] = frame[v][u]
    print(output.shape)
    return output


def preprocess(frames, raw):
    return frames[0]

if __name__ == '__main__':
    parser = make_parser()
    camera, args = make_camera_with_args(parser=parser, log=False, fps=15, res=(1280, 720))
    camera.stream(prepare=prepare, preprocess=preprocess, frames_stored=1)