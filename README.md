## PyTorch Pretrained BERT for Named Entity Recognition
This is a conversion of the NER code from [BioBERT](https://github.com/dmis-lab/biobert) to Pytorch using [Pytorch Pretrained BERT](https://github.com/huggingface/pytorch-pretrained-BERT) from Hugging Face.

It can be used to train an NER model on datasets such as those referenced on the BioBERT github page (these are datasets such as the NCBI disease dataset that have been preprocessed for the NER task.)

### Instructions
1. CLone the repo
2. `cd pytorch-pretrained-BERT-ner`
3. `pip install -r requirements.txt`
4. `python run_ner.py --do_train --do_eval --do_lower_case --data_dir /path/to/dataset --bert_model /path/to/bert/model --max_seq_length 128 --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir /tmp/bert_output`

Where /path/to/bert/model is a directory containing a pytorch BERT model (i.e. one that was converted using the conversion script [here](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/convert_pytorch_checkpoint_to_tf.py)) and path/to/dataset is a directory containing train.tsv and dev.tsv files in the proper format(see BioBERT).
