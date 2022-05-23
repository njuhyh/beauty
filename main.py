import argparse
import sys
import works
import os.path
from works import SUPPORTED_MODELS, alpha_cover, generate_accesory
import cv2
import numpy as np
import traceback

THINNING_FORCE = 0.3

def run_process():
    parser = argparse.ArgumentParser(description='Beautify image or video. Large images may take a long time.')
    parser.add_argument('src', type=str, help='source file path')
    parser.add_argument('-i', '--image', action='store_true', help='process an image')
    parser.add_argument('-v', '--video', action='store_true', help='process a video')
    parser.add_argument('-k', '--keep', action='store_true', help='keep the face raw. No beautification')
    parser.add_argument('-m', '--mesh', action='store_true', help='generate face mesh')
    parser.add_argument('-c', '--compare', action='store_true', help='concatenate the original image and the processed one, for comparison')
    parser.add_argument('-r', '--realtime', action='store_true', help='when processing video, show realtime result in a window')
    parser.add_argument('-a', '--accessory', type=str, choices=SUPPORTED_MODELS.keys(), help=f'generate an accessory. Supported: {SUPPORTED_MODELS.keys()}')
    parser.add_argument('-o', '--output', help='specify output file name. For image use .png or .jpg. For video use .mp4 only.')
    args = parser.parse_args()

    if args.image and args.video:
        print('Cannot choose both image and video modes.', file=sys.stderr)
        exit()

    if not args.image and not args.video:
        print('Choose one in image (using -i) and video (using -v) mode.', file=sys.stderr)
        exit()

    try:
        assert os.path.isfile(args.src)
    except Exception as err:
        print('Source not found.', file=sys.stderr)
        exit()

    def process(image):
        image_ = image.copy()
        mesh_result = works.generate_mesh(works.face_mesh, image)
        if mesh_result.multi_face_landmarks is None:
            return image
        landmarks = mesh_result.multi_face_landmarks[0].landmark
        if not args.keep:
            image = works.face_whitening(image, landmarks)
            image = works.face_thinning(image, landmarks, THINNING_FORCE)
        if args.mesh:
            works.draw_mesh(image, mesh_result)
        if args.accessory:
            acc = generate_accesory(mesh_result, image.shape, args.accessory)
            image = alpha_cover(acc, image)
        if args.compare:
            image = np.concatenate([image_, image], axis=1)
        return image

    try:
        works.init()
    except Exception as err:
        print('Generator initialization error.', file=sys.stderr)
        exit()

    if args.image:
        image = None
        try:
            image = cv2.imread(args.src)
        except Exception as err:
            print('Cannot read source image.', file=sys.stderr)
            exit()
        image = process(image)

        out_path = None
        if args.output:
            out_path = args.output
        else:
            out_path = os.path.basename(os.path.splitext(args.src)[0]) + '_out.png'
        
        try:
            cv2.imwrite(out_path, image)
        except Exception as err:
            print('Error writing to output.', file=sys.stderr)
            exit()
    else:
        cap = None
        out_path = None
        if args.output:
            out_path = args.output
        else:
            out_path = os.path.basename(os.path.splitext(args.src)[0]) + '_out.mp4'
        out = None
        try:
            cap = cv2.VideoCapture(args.src)
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)*(2 if args.compare else 1)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(w), int(h)))
            assert cap.isOpened()
        except Exception as err:
            print('Cannot read source video or output file path.', file=sys.stderr)
            exit()

        try:
            i = 0
            while cap.isOpened() and out.isOpened():
                ret, frame = cap.read()
                if ret:
                    print('rendering frame %d    ' % i, end='\r')
                    i += 1
                    frame = process(frame)
                    if args.realtime:
                        cv2.imshow('frame', frame)
                    out.write(frame)
                else:
                    break
        except Exception as err:
            print('Video processing error. Details:\n', traceback.format_exc(), file=sys.stderr)
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()    

        works.base.destroy()
    exit()

if __name__ == '__main__':
    run_process()