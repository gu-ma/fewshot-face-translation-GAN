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
    src, mask, _, _, _ = utils.get_src_inputs(source, fd, fp, idet)
    tar, emb_tar = utils.get_tar_inputs([target], fd, fv)
    out = generator.inference(src, mask, tar, emb_tar)
    result_face = np.squeeze(((out[0] + 1) * 255 / 2).astype(np.uint8))
    return result_face

if __name__ == "__main__":
    runway.run()
