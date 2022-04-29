import tensorflow as tf
from transformers import TFT5ForConditionalGeneration
from google.colab import files

# To make sure Google Colab's environment does not kill the script, we assign a custom memory limit
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)

# Here the T5 model is loaded from HuggingFace library
model = TFT5ForConditionalGeneration.from_pretrained('Vamsi/T5_Paraphrase_Paws')

# Here we assign two input ids which are supposed to be the sentence entered to be paraphrased padded to the max length, which is 256 words long.
# Words are represented by integers
input_spec = {
    'decoder_input_ids': tf.TensorSpec([1, 256], tf.int32, name="decoder_input_ids"),
    'input_ids': tf.TensorSpec([1, 256], tf.int32, name="input_ids"),
    'attention_mask': tf.TensorSpec([1, 256], tf.int32, name="attention_masks"),
}
model._saved_model_inputs_spec = None
model._set_save_spec(input_spec)

# The converter is started with the from_keras_model() function
converter = tf.lite.TFLiteConverter.from_keras_model(model);
# The default optimization in size is used
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Here we define the supported operations
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert();

model_path = './models/t5_paraphrase_paws-256.tflite'

with open(model_path, 'wb') as o_:
    o_.write(tflite_model)

files.download(model_path)
