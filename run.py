from asr_models.sensevoice_model import SenseVoiceASRModel
from asr_models.whisper_model import WhisperASRModel
from asr_datasets.common_voice import CommonVoiceDataset
from asr_datasets.guangzhou_daily_use import GuangzhouDailyUseDataset
from asr_datasets.guangzhou_cabin import GuangzhouCabinDataset
import torch
import librosa
import json
import os
from tqdm import tqdm

device = ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
batch_size = 64
num_models = 2
num_datasets = 3

for dataset_index in range(num_datasets):
    if dataset_index == 0:
        dataset = CommonVoiceDataset(batch_size=batch_size)
    elif dataset_index == 1:
        dataset = GuangzhouDailyUseDataset(batch_size=batch_size)
    elif dataset_index == 2:
        dataset = GuangzhouCabinDataset(batch_size=batch_size)
    
    dataset_name = dataset.get_name()

    for model_index in range(num_models):
        if model_index == 0:
            model = SenseVoiceASRModel(device=device)
        elif model_index == 1:
            model = WhisperASRModel(device=device)
        
        model_name = model.get_name()
        results = []
        for batch_audios, batch_sentences in tqdm(dataset):
            transcriptions = model.generate([
                librosa.resample(audio['array'], orig_sr=audio['sampling_rate'], target_sr=16000)
                for audio in batch_audios
            ])
            for transcription, sentence in zip(transcriptions, batch_sentences):
                results.append({"transcription": transcription["text"], "expected": sentence})
            break

        # Create directory if it doesn't exist
        os.makedirs(f'results/{model_name}', exist_ok=True)
        
        # Save results to a JSON file
        with open(f'results/{model_name}/{dataset_name}.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
