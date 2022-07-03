## BERTopic-inference

This repository contains some starting pointers for implementing your own BERTopic inference server using NVIDIA's Triton.

There are three models: `BERTopic-UMAP` that's GPU intensive and needs to run on that kind of host, `BERTopic-HDBSCAN` that should run on CPUs and `BERTopic-inference` that merges the two in the same server if you can't distribute the load better. 

## Packing Python

Triton needs your Python enviroment, and due to licensing I cannot upload mine. Here is how you can make and upload your own:

```
# conda create -n rapids-triton -c rapidsai -c nvidia -c conda-forge rapids=22.06 python=3.8 cudatoolkit=11.5
# run export PYTHONNOUSERSITE=True BEFORE CONDA ACTIVATE rapids-triton
# pip install hdbscan pympler psutil 
# conda install conda-pack
# conda-pack, move result to root of BERTopic-deploy folder as rapids-triton.tar.gz
```

This assumes you're using the same Python versions between your enviroment and Triton. Look at Triton documentation if they are different (or just change the conda create command above to use the same Python version your Triton ships with).

## Saving your BERTopic model

This compresses your BERTopic model down to it's composing models to save on everything. Run it after training.

```
fl = open('/root/hdbscan.pickle','wb')
pickle.dump(your_bertopic_model.hdbscan_model, fl)
fl.close()

fl = open('/root/umap.pickle','wb')
pickle.dump(your_bertopic_model.umap_model, fl)
fl.close()

fl = open('/root/mappings.pickle','wb')
pickle.dump(your_bertopic_model.topic_mapper.get_mappings(original_topics=True), fl)
fl.close()
```


## Curl examples

Getting started with Triton can be overwhelming if you've never used it before. Run those commands if you want to get a feel for things: 

```
curl -X POST  http://localhost:8000/v2/models/BERTopic-inference/versions/1/infer --data-binary "@test_query_inference"  --header "Inference-Header-Content-Length: 161"
curl -X POST  http://localhost:8000/v2/models/BERTopic-umap/versions/1/infer --data-binary "@test_query_umap"  --header "Inference-Header-Content-Length: 165"
```

If you open the test_query files you'll find ascii character rather than what the model truly expects: FP32 binary bytes. Yes, you need to run your BERT embeddings separately and send them to the model in binary form, **it doesn't take ASCII input!**