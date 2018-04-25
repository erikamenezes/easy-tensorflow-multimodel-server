import tensorflow as tf
import argparse 
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import json
import os
from flask import Flask, redirect, request, Response, flash
from werkzeug.utils import secure_filename
import glob

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
MODEL_SUFFIX = '.pb'
LABEL_MAP_SUFFIX = '.pbtxt'

MODEL_FOLDER = os.getenv('MODEL_FOLDER', './models_classify')
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './pics/')


PORT = int(os.getenv('PORT', '5432'))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_labelmap(filename):
    categories = []
    for category in open(filename, "r"):
        categories.append(category)
    return categories

def create_category_index(categories):
  """Creates dictionary of COCO compatible categories keyed by category id.

  Args:
    categories: a list of dicts, each of which has the following keys:
      'id': (required) an integer id uniquely identifying this category.
      'name': (required) string representing category name
        e.g., 'cat', 'dog', 'pizza'.

  Returns:
    category_index: a dict containing the same entries as categories, but keyed
      by the 'id' field of each category.
  """
  category_index = {}
  for i,cat in enumerate(categories):
    category_index[i] = cat
  return category_index

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def load_model(model_dir, model_prefix):
    categories = load_labelmap('{}/{}{}'.format(model_dir, model_prefix, LABEL_MAP_SUFFIX))
    category_index = create_category_index(categories)

    with tf.Graph().as_default() as classification_graph:
        ic_graph_def = tf.GraphDef()
   
        with tf.gfile.GFile('{}/{}{}'.format(model_dir, model_prefix, MODEL_SUFFIX), "rb") as f:
            
            ic_graph_def.ParseFromString(f.read())
            tf.import_graph_def(ic_graph_def, name='')

            ops = classification_graph.get_operations()
            all_tensor_names = {
                    output.name
                    for op in ops for output in op.outputs
                }

            tensor_dict = {}
            for key in [
                        'loss', 
                ]:
                tensor_name = key + ':0'
                
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = classification_graph.get_tensor_by_name(tensor_name)
                image_tensor = classification_graph.get_tensor_by_name('Placeholder:0')
                sess = tf.Session(graph=classification_graph)
            print("tensor_dict", tensor_dict)

    return {
        'session': sess,
        'image_tensor': image_tensor, 
        'tensor_dict': tensor_dict,
        'category_index': category_index
    }


def load_models(model_dir):

    models = {}
    for model_file in glob.glob('{}/*{}'.format(model_dir, MODEL_SUFFIX)):
        model_prefix = os.path.basename(model_file)[:-len(MODEL_SUFFIX)]
        print('Loading model {} from {}/{}{}'.format(model_prefix, model_dir, model_prefix, MODEL_SUFFIX))
        models[model_prefix] = load_model(model_dir, model_prefix)
    return models

def evaluate(model, filename):

    image = cv2.imread(filename)
    resized_image = cv2.resize(image, (227, 227)) 

    image_np = np.asarray(resized_image)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    output_dict = model['session'].run(
        model['tensor_dict'], feed_dict={model['image_tensor']: image_np_expanded})


    result_idx = np.argmax(output_dict['loss'])
    print(result_idx)
    result = {}
    print(model['category_index'])
    result['class'] = model['category_index'][result_idx]
    print(result)
    return (json.dumps(result))


@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        if 'file' not in request.files:
            return Response(response='Missing file', status=400)
        if 'modelname' not in request.form:
            return Response(response='Missing modelname', status=400)
        modelname = request.form['modelname']
        if modelname not in app.config['MODELS']:
            models = load_models(MODEL_FOLDER)
            app.config['MODELS'] = models
            if modelname not in app.config['MODELS']:
                return Response(response='Model {} not found'.format(modelname), status=404)
        
        model = app.config['MODELS'][modelname]
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                print('Evaluating {} with model {}'.format(filepath, modelname))
                response = Response(response=evaluate(model, filepath), status=200, mimetype='application/json')
            except Exception as e:
                print(e)
                response = Response(response=str(e), status=501)
            os.remove(filepath)
            return response
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p>
      <input type=text name=modelname>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

def teardown(models):
    for model in models:
        print('Tearing down {}'.format(model))
        models[model]['session'].close()
        
import atexit
if __name__ == '__main__':
    
    models = load_models(MODEL_FOLDER)
    print('Loading models')
    atexit.register(lambda: teardown(models))
    app.config['MODELS'] = models
    app.run(host='0.0.0.0', port=PORT, debug=False)

