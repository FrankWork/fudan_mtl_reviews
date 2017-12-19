## Adversarial Multi-task Learning for Text Classification

TensorFlow implementation of the paper [Adversarial Multi-task Learning for Text Classification](http://www.aclweb.org/anthology/P17-1001). 

- The code uses CNN instead of LSTM. 
- The Gradient Reversal Layer is copied from https://github.com/pumpikano/tf-dann. 
- The Orthogonality Constraints loss (diff loss) is copied from 'research/domain_adaptation' of https://github.com/tensorflow/models. The correlation matrix is normalized. Otherwise, the loss value will be too large.

performace using 50d word embedding:
models           | avg error
-----------------|-----------
mtl              | 13.75
mtl + adv        | 12.79
mtl + adv + diff | 12.70

To train the model:
```
cd data/
tar zxvf fudan-mtl-dataset.tar.gz
cd ../
python3 src/main.py --build_data
python3 src/main.py --adv
```