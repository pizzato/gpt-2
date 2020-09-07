#!/usr/bin/env python3

# import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder
import streamlit as st



def interact_model(
    model_name='117M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=0.0,
    raw_text='<|endoftext|>'
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    """
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)


        context_tokens = enc.encode(raw_text)
        generated = 0
        for _ in range(nsamples // batch_size):
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
            for i in range(batch_size):
                generated += 1
                text = enc.decode(out[i])
                if raw_text != "<|endoftext|>":
                    text = raw_text + ' ' + text

                text = text.replace("<|endoftext|>","-----")
                st.markdown(text)


if __name__ == '__main__':
    #fire.Fire(interact_model)

    model_name = st.sidebar.selectbox('Model Name',('PawPatrol','117M'))

    seed = st.sidebar.number_input("seed", value=0)
    if seed == 0:
        seed = None

    nsamples = st.sidebar.number_input("nsamples", min_value=1, max_value=5, value=1)
    batch_size = st.sidebar.number_input("batch_size", min_value=1, max_value=10, value=1)
    length = st.sidebar.number_input("length", min_value=0, max_value=150, value=0)
    if length == 0:
        length = None

    temperature = st.sidebar.slider("temperature", min_value=0.0, max_value=1.0, value=0.90, step=0.01)
    top_k = st.sidebar.slider("top_k", min_value=0, max_value=100, value=0)
    top_p = st.sidebar.slider("top_p", min_value=0.0, max_value=5.0, value=0.0, step=0.01)

    raw_text = st.text_input("Paw Patrol Title or Empty", value="")
    if raw_text == "":
        raw_text = "<|endoftext|>"

    if st.button("Generate"):
        interact_model(model_name=model_name,
                       seed=seed,
                       nsamples=nsamples,
                       batch_size=batch_size,
                       length=length,
                       temperature=temperature,
                       top_k=top_k,
                       top_p=top_p,
                       raw_text=raw_text
                       )