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
    shutil.move(opts['encoder'], 'weights/encoder.h5')
    shutil.move(opts['decoder'], 'weights/decoder.h5')
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

@runway.command('translate', inputs={'source': runway.image, 'target': runway.image}, outputs={'result': runway.image})
def translate(models, inputs):
    fd = models['fd']
    fp = models['fp']
    idet = models['idet']
    fv = models['fv']
    generator = models['generator']
    source = np.array(inputs['source'])
    target = np.array(inputs['target'])
    src, mask, aligned_im, (x0, y0, x1, y1), landmarks = utils.get_src_inputs(source, fd, fp, idet)
    tar, emb_tar = utils.get_tar_inputs([target], fd, fv)
    out = generator.inference(src, mask, tar, emb_tar)
    result_face = np.squeeze(((out[0] + 1) * 255 / 2).astype(np.uint8))
    result_img = utils.post_process_result(source, fd, result_face, aligned_im, src, x0, y0, x1, y1, landmarks)
    return result_img

if __name__ == "__main__":
    runway.run()
