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
    model = FaceTranslationGANInferenceModel()
    fv = FaceVerifier(classes=512)
    fp = face_parser.FaceParser()
    fd = face_detector.FaceAlignmentDetector()
    idet = IrisDetector()
    return {
        'model': model,
        'fp': fp,
        'fd': fd,
        'fv': fv,
        'idet': idet
    }

@runway.command('translate', inputs={'source': runway.image, 'target': runway.image}, outputs={'result': runway.image})
def translate(model, inputs):
    fd = model['fd']
    fp = model['fp']
    idet = model['idet']
    fv = model['fv']
    source = np.array(inputs['source'])
    target = np.array(inputs['target'])
    src, mask, _, _, _ = utils.get_src_inputs(source, fd, fp, idet)
    tar, emb_tar = utils.get_tar_inputs([target], fd, fv)
    out = model.inference(src, mask, tar, emb_tar)
    return out

if __name__ == "__main__":
    runway.run()
