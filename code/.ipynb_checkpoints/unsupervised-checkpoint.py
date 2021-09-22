import csv
import os
import pandas as pd
import boto3
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses
from ipywidgets import widgets
import random
import logging
import sys
import argparse
import torch
import nltk
nltk.download('punkt')

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)

    # Data, model, and output directories
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    word_embedding_model = models.Transformer(args.model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=False, 
                                   pooling_mode_cls_token = True)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    # read data
    print('******* list files *********: ', os.listdir(args.training_dir))
    print('********************** Reading Data *************************')
    df_data = pd.read_csv(os.path.join(args.training_dir, 'train.csv'), index_col=0)
    print(df_data.head(5))
    train_sentences = df_data['Description'].tolist()
    
    train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    #use the denoising auto-encoder loss
    train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=args.model_name, tie_encoder_decoder=True)

    #call the fit method
    model.fit(train_objectives=[(train_dataloader, train_loss)],
             epochs = args.epochs, 
             weight_decay=0, 
             scheduler='constantlr', 
             optimizer_params={'lr': args.learning_rate}, 
             )
    model.save(args.model_dir)