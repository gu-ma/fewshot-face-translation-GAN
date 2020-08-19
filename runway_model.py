from models import FaceTranslationGANInferenceModel
from face_toolbox_keras.models.detector.iris_detector import IrisDetector
from face_toolbox_keras.models.verifier.face_verifier import FaceVerifier
from face_toolbox_keras.models.parser import face_parser
from face_toolbox_keras.models.detector import face_detector
from utils import utils
import runway
import numpy as np
import shutil

@runway.setup(options={'encoder': runway.file(extension='.h5'), 'decoder': runway.file(extension='.h5')})
def setup(opts):
    shutil.copy(opts['encoder'], 'weights/encoder.h5')
    shutil.copy(opts['decoder'], 'weights/decoder.h5')
    generator = FaceTranslationGANInferenceModel()
    fv = FaceVerifier(classes=512)
    fp = face_parser.FaceParser()
    fd = face_detector.FaceAlignmentDetector()
    fd.build_FAN()
    idet = IrisDetector()
    return {
        'generator': generator,
        'fp': fp,
        'fd': fd,
        'fv': fv,
        'idet': idet
    }

last_target = None
aligned_target = None
target_embedding = None

@runway.command('translate', inputs={'source': runway.image, 'target': runway.image}, outputs={'result': runway.image})
def translate(models, inputs):
    global aligned_target
    global target_embedding
    global last_target
    fd = models['fd']
    fp = models['fp']
    idet = models['idet']
    fv = models['fv']
    generator = models['generator']
    source = np.array(inputs['source'])
    target = np.array(inputs['target'])
    src, mask, aligned_im, (x0, y0, x1, y1), landmarks = utils.get_src_inputs(source, fd, fp, idet)
    if last_target is None or not np.array_equal(last_target, target):
        aligned_target, target_embedding = utils.get_tar_inputs([target], fd, fv)
    last_target = target
    out = generator.inference(src, mask, aligned_target, target_embedding)
    result_face = np.squeeze(((out[0] + 1) * 255 / 2).astype(np.uint8))
    result_img = utils.post_process_result(source, fd, result_face, aligned_im, src, x0, y0, x1, y1, landmarks)
    return result_img

if __name__ == "__main__":
    runway.run()
