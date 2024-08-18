import sys

import soundfile as sf
import logging
import os
import io
import torch
import glob
import uuid

from fastapi import FastAPI, Response
from pydantic import BaseModel

from frontend import g2p_cn_en, ROOT_DIR, read_lexicon, G2p
from models.prompt_tts_modified.jets import JETSGenerator
from models.prompt_tts_modified.simbert import StyleEncoder
from transformers import AutoTokenizer
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from yacs import config as CONFIG
from config import Config
from collections import defaultdict
import time

from torch.profiler import profile, record_function, ProfilerActivity

torch.set_float32_matmul_precision('high')

LOGGER = logging.getLogger(__name__)
ROOT_DIR = "."

DEFAULTS = {
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
config = Config()
MAX_WAV_VALUE = 32768.0

def get_env(key):
    return os.environ.get(key, DEFAULTS.get(key))

def get_int_env(key):
    return int(get_env(key))

def get_float_env(key):
    return float(get_env(key))

def get_bool_env(key):
    return get_env(key).lower() == 'true'

def scan_checkpoint(cp_dir, prefix, c=8):
    pattern = os.path.join(cp_dir, prefix + '?'*c)
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def get_models():

    am_checkpoint_path = scan_checkpoint(
        f'{config.output_directory}/prompt_tts_open_source_joint/ckpt', 'g_')

    # f'{config.output_directory}/style_encoder/ckpt/checkpoint_163431'
    style_encoder_checkpoint_path = scan_checkpoint(
        f'{config.output_directory}/style_encoder/ckpt', 'checkpoint_', 6)

    with open(config.model_config_path, 'r') as fin:
        conf = CONFIG.load_cfg(fin)

    conf.n_vocab = config.n_symbols
    conf.n_speaker = config.speaker_n_labels

    style_encoder = StyleEncoder(config)
    model_CKPT = torch.load(style_encoder_checkpoint_path, map_location=torch.device("cuda"))
    model_ckpt = {}
    for key, value in model_CKPT['model'].items():
        new_key = key[7:]
        model_ckpt[new_key] = value
    style_encoder.load_state_dict(model_ckpt, strict=False)
    style_encoder.eval()
    style_encoder = style_encoder.cuda()
    generator = JETSGenerator(conf).to(DEVICE)

    model_CKPT = torch.load(am_checkpoint_path, map_location=DEVICE)
    generator.load_state_dict(model_CKPT['generator'])

    #torch._dynamo.config.capture_scalar_outputs = True
    #generator = torch.compile(generator, fullgraph=True, dynamic=True)
    generator.eval()
    dummy_output = generator(
        inputs_ling=torch.ones(1, 100, device="cuda").long(),
        inputs_style_embedding=torch.rand(1, 768, device="cuda"),
        input_lengths=torch.tensor([100], device="cuda").long(),
        inputs_content_embedding=torch.rand(1, 768, device="cuda"),
        inputs_speaker=torch.tensor([0], device="cuda").long(),
        alpha=1.0
    )
    print("Compiled model!")      

    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)

    with open(config.token_list_path, 'r') as f:
        token2id = {t.strip(): idx for idx, t, in enumerate(f.readlines())}

    with open(config.speaker2id_path, encoding='utf-8') as f:
        speaker2id = {t.strip(): idx for idx, t in enumerate(f.readlines())}

    return (style_encoder, generator, tokenizer, token2id, speaker2id)

def get_style_embedding(prompt, tokenizer, style_encoder):
    start_time = time.time()
    prompt = tokenizer([prompt], return_tensors="pt")

    input_ids = prompt["input_ids"].to('cuda')
    token_type_ids = prompt["token_type_ids"].to('cuda')
    attention_mask = prompt["attention_mask"].to('cuda')

    with torch.no_grad():
        start_time = time.time()
        output = style_encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        # print(f"Style encoder inference took {time.time() - start_time:.4f} seconds")

    start_time = time.time()
    style_embedding = output["pooled_output"].cpu().squeeze().numpy()
    # print(f"Converting output to numpy took {time.time() - start_time:.4f} seconds")

    return style_embedding

def emotivoice_tts(text, prompt, content, speaker, models):
    (style_encoder, generator, tokenizer, token2id, speaker2id) = models
    # print(' ')
    start_time = time.time()
    style_embedding = get_style_embedding(prompt, tokenizer, style_encoder)
    print(f"Time taken for getting style embedding: {time.time() - start_time:.4f} seconds")

    start_time = time.time()
    content_embedding = get_style_embedding(content, tokenizer, style_encoder)
    print(f"Time taken for getting content embedding: {time.time() - start_time:.4f} seconds")

    start_time = time.time()
    speaker = speaker2id[speaker]
    # print(f"Time taken for getting speaker ID: {time.time() - start_time:.4f} seconds")

    start_time = time.time()
    text_int = [token2id[ph] for ph in text.split()]
    # print(f"Time taken for converting text to integer tokens: {time.time() - start_time:.4f} seconds")

    start_time = time.time()
    sequence = torch.from_numpy(np.array(text_int)).to(DEVICE).long().unsqueeze(0)
    # print(f"Time taken for creating sequence tensor: {time.time() - start_time:.4f} seconds")

    start_time = time.time()
    sequence_len = torch.from_numpy(np.array([len(text_int)])).to(DEVICE)
    style_embedding = torch.from_numpy(style_embedding).to(DEVICE).unsqueeze(0)
    # print(f"Time taken for creating sequence length tensor and moving style embedding to device: {time.time() - start_time:.4f} seconds")

    start_time = time.time()
    content_embedding = torch.from_numpy(content_embedding).to(DEVICE).unsqueeze(0)
    speaker = torch.from_numpy(np.array([speaker])).to(DEVICE)
    # print(f"Time taken for creating content embedding and speaker tensor: {time.time() - start_time:.4f} seconds")

    with torch.inference_mode():
        start_time = time.time()
        infer_output = generator(
            inputs_ling=sequence,
            inputs_style_embedding=style_embedding,
            input_lengths=sequence_len,
            inputs_content_embedding=content_embedding,
            inputs_speaker=speaker,
            alpha=1.0
        )
        print(f"Time taken for generator inference: {time.time() - start_time:.4f} seconds")

    start_time = time.time()
    audio = infer_output["wav_predictions"].squeeze() * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype('int16')
    # print(f"Time taken for processing audio output: {time.time() - start_time:.4f} seconds")

    return audio

def generate_unique_filename(extension):
    unique_id = str(uuid.uuid4())
    return f"{unique_id}.{extension}"

def save_audio_file(data, filename):
    with open(filename, 'wb') as file:
        file.write(data)

speakers = config.speakers
(style_encoder, generator, tokenizer, token2id, speaker2id) = get_models()
models = (style_encoder, generator, tokenizer, token2id, speaker2id)
lexicon = read_lexicon(f"{ROOT_DIR}/lexicon/librispeech-lexicon.txt")
g2p = G2p()

def get_audio(input_text):
    start_time = time.time()
    phonemized_text = g2p_cn_en(input_text, g2p, lexicon)
    # print('phonemized_text took', time.time() - start_time)

    start_time = time.time()
    np_audio = emotivoice_tts(phonemized_text, '', input_text, '1088', models)
    print('emotivoice_tts took', time.time() - start_time)
        
    return np_audio

get_audio('asdfasd mad发多少l')

with open('examples.txt', 'r') as f:
    sentences = f.readlines()

times = []
from tqdm import tqdm
print('')
for sentence in sentences:
    print()
    start_time = time.time()
    np_audio = get_audio(sentence)
    end_time = time.time()

    time_taken = end_time - start_time
    times.append(time_taken)
    print(f"Get_audio took {time_taken:.2f} seconds")

print(np.array(times).mean())

wav_buffer = io.BytesIO()
sf.write(file=wav_buffer, data=np_audio,
          samplerate=16000, format='WAV')
buffer = wav_buffer

# Generate a unique filename for the audio file
file_path = 'audio.wav'

save_audio_file(wav_buffer.getvalue(), file_path)