import json
import sys
import os
import libspacy
import libgrams
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn.utils import shuffle
#from sklearn.manifold import TSNE
from tqdm import tqdm
from only_ascii import clean_str
#import libglove


def main():
  train_dir1 = 'clickbait17-train-170331'
  instances_filename = 'instances.jsonl'
  truths_filename = 'truth.jsonl'
  train_dir2 = 'clickbait17-train-170630'
  instances_filename = 'instances.jsonl'
  truths_filename = 'truth.jsonl'
  raw_data={} #Data set indexed by the id, id is a string
  raw_truths={} #Truths indexed by the id, id is a string


  abs_path = os.path.join(train_dir1, instances_filename)
  fp = open(abs_path)
  for line in fp:
    json_obj = json.loads(line)
    #print json_obj['postText']
    item_id = json_obj['id']
    raw_data[item_id]=json_obj

  #print raw_data

  fp.close()
  abs_path = os.path.join(train_dir2, instances_filename)
  fp = open(abs_path)
  for line in fp:
    json_obj = json.loads(line)
    #print json_obj['postText']
    item_id = json_obj['id']
    raw_data[item_id]=json_obj


  abs_path = os.path.join(train_dir1, truths_filename)
  fp = open(abs_path)
  for line in fp:
    json_obj = json.loads(line)
    item_id = json_obj['id']
    raw_truths[item_id]=json_obj

  abs_path = os.path.join(train_dir2, truths_filename)
  fp = open(abs_path)
  for line in fp:
    json_obj = json.loads(line)
    item_id = json_obj['id']
    raw_truths[item_id]=json_obj

  feature = 'postText'

  raw_cb = []
  y_cb=[]
  y_labels=[]
  item_ids=[]
  cb=[]
  nocb=[]
  item_ids_cb=[]
  item_ids_nocb=[]
  for item_id in tqdm(raw_data):
    rating = raw_truths[item_id]['truthMean']
    postText = raw_data[item_id].get('postText','')
    postText = clean_str(postText[0].strip())
    label=raw_truths[item_id]['truthClass']
    #print postText
    #sys.exit()
    targetTitle = clean_str(raw_data[item_id].get('targetTitle',''))
    targetDescription = clean_str((raw_data[item_id].get('targetDescription','')))
    targetKeywords = ' '.join(raw_data[item_id].get('targetKeywords','').split(','))
    targetParagraphs = clean_str(str(raw_data[item_id].get('targetParagraphs','')))
    targetCaptions = clean_str(str(raw_data[item_id].get('targetCaptions','')))

    content = postText
    item = raw_data[item_id]
    item['rating'] = rating
    item['postText']=postText
    #content = clean_str(postText + targetTitle + targetDescription + targetKeywords + targetParagraphs + targetCaptions)
    #print "Content=",content
    #print item_id, label
    raw_cb.append(item)
    if label=='clickbait':
      cb.append(item)
      item_ids_cb.append(item_id)
    else:
      nocb.append(item)
      item_ids_nocb.append(item_id)
    label = rating_to_class(rating)
    y_cb.append(rating)
    y_labels.append(label)
    item_ids.append(item_id)
  fp.close()

  cb=shuffle(cb, random_state=0)
  nocb=shuffle(nocb, random_state=0)
  nocb=nocb[:len(cb)]
  item_ids_nocb=item_ids_nocb[:len(cb)]
  print "CB=", len(cb), "NOCB=", len(nocb), "ITEM_IDS_CB", len(item_ids_cb), "ITEM_IDS_NOCB", len(item_ids_nocb)
  y_cb = [0]*len(nocb)+[1]*len(cb)
  raw_cb = nocb + cb
  item_ids = item_ids_nocb + item_ids_cb
  (raw_cb, y_cb, item_ids) = shuffle(raw_cb, y_cb, item_ids, random_state=0)


  #(raw_cb, y_cb, item_ids) = shuffle(raw_cb, y_cb, item_ids, random_state=0)

  #create the dataset for subba
  fa=open('clickbait_titles.txt','w')
  fb=open('clickbait_ratings.txt','w')
  for (cb, rating) in zip(raw_cb, y_cb):
    fa.write(cb['postText']+'\n')
    fb.write(str(rating)+'\n')
  fa.close()
  fb.close()


  #(X, Y) = make_scatter(raw_cb, y_cb)
  train_percent=0.8
  train_size=int(len(raw_cb)*train_percent)
  X_raw_train = raw_cb[:train_size]
  y_train = y_cb[:train_size]
  train_ids = item_ids[:train_size]


  X_raw_test = raw_cb[train_size:]
  y_test = y_cb[train_size:]
  test_ids = item_ids[train_size:]

  #create test annotations
  fp=open("test_annotations.jsonl",'w')
  for item_id in test_ids:
    json_obj = raw_truths[item_id]
    json_str = json.dumps(json_obj)
    fp.write(json_str+'\n')
  fp.close()


  print "X_raw_train, y_train", len(X_raw_train), len(y_train)
  print "X_raw_test, y_test", len(X_raw_test), len(y_test) 
  X_train=[]
  X_test=[]
  print("Extracting features from train")
  for item in tqdm(X_raw_train):
    raw_title = item['postText']
    c_title = clean_title(raw_title)
    #print raw_title, c_title
    features = generate_features(raw_title, c_title)
    #print(features)
    #sys.exit()
    X_train.append(features)

  print( "Extracting features from test")
  for item in tqdm(X_raw_test):
    raw_title = item['postText']
    c_title = clean_title(raw_title)
    features = generate_features(raw_title, c_title)
    X_test.append(features)

  num_features = len(features)
  print( "Size of train, test", len(X_train), len(X_test))
  print( "Size of  labels train, test", len(y_train), len(y_test))
  print( "#features=", num_features)


  print("Try linear regression")
  model = linear_model.LinearRegression()
  #model = svm.SVR(C=1.0, epsilon=0.2)
  model.fit(X_train, y_train)
  print("Mean squared error test: %.4f" % np.mean((model.predict(X_test) - y_test) ** 2))
  print("Mean squared error train: %.4f" % np.mean((model.predict(X_train) - y_train) ** 2))

  print("Minor improvements")
  y_pred = model.predict(X_test)
  y_pred = [ 0 if i < 0 else i for  i in y_pred]
  y_pred = [ 1 if i > 1 else i for  i in y_pred]
  y_pred = np.array(y_pred)
  print("Mean squared error test: %.4f" % np.mean((y_pred - y_test) ** 2))
  #print y_pred
  #y_pred = np.random.rand(len(X_test)) #Uncomment this line to check with random guesses
  create_predictions("test_predictions", y_pred, test_ids)
  create_predictions("test_truths", y_test, test_ids)
  #Print those instances where the prediction varies by a threshold
  
  fp=open('max_errors.txt','w')
  for (y_p, y_real, test_id) in zip(y_pred, y_test, test_ids):
    if abs(y_p - y_real) > 0.4:
      line ='%s %f %f' % (raw_data[test_id][feature], y_p, y_real)
      fp.write(line+'\n')
  fp.close()
  #print(model.coef_)
  #print(sorted(model.coef_.tolist()))
  os.system('python eval.py test_annotations.jsonl test_predictions outfile')
  print model.coef_, model.intercept_

#End of main
def generate_features(title, c_title):
  features=[]
  #title=clean_str(title.decode('utf-8'))
  #print title
  vecs = libspacy.get_vector(title)


  return vecs.tolist()


def create_raw_file(filename, data):
  fd = open(filename,'w')
  for row in data:
    fd.write(row+'\n')
  fd.close()

def rating_to_class(rating):
  if rating > 0.75:
    return 3
  if rating > 0.5:
    return 2
  if rating > 0.25:
    return 1
  return 0

def create_predictions(filename, predictions, item_ids):
  fd = open(filename, 'w')
  for (item_id, prediction) in zip(item_ids, predictions):
    json_obj={"id":item_id, "clickbaitScore":float(prediction)}
    json_str = json.dumps(json_obj)
    fd.write(json_str+'\n')

  fd.close()

def clean_title(title):
  title=title.replace("'", " ")
  title=title.replace("  ", " ")
  words = title.lower().split(' ')
  words = [ w for w in words if not w.startswith('@')]
  words = [ w for w in words if not w.startswith('#')]
  words = [ w for w in words if not w.startswith('rt')]
  #words = [ w for w in words if len(w) > 1 and not w[0].isdigit()]

  return ' '.join(words)

if __name__ == "__main__":
  main()
