# SentEval + torchst

pytorch-skipthoughts evaluation code for SentEval. Use this script to evaluate models trained using torchst on 
various supervised tasks.

Following files or conditions must be prepared first:

  - `pytorch-skipthoguhts` installation and model training: refer to https://github.com/kaniblu/pytorch-skipthoughts for more info.
  - vectorizer configuration file: a configuration file that can be supplied to `torchst.vectorizer`.
  - senteval data: datasets that are downloaded using `get_transfer_data_ptb.bash` script provided in `SentEval` repository..
  - `senteval` package: the package must be installed using `python setup.py install` command.
  - model file: of course, we also need a checkpoint file to test on.

Then run the following command to supply above files to the evaluation script:

```bash
    pip install -r requirements.txt
    python eval.py --vectorizer-config <path-to-vectorizer-yml> --senteval-data <path-to-senteval-data> --batch-size <batch-size> --config tasks-all.yml
```
