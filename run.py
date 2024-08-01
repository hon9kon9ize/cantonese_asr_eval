from asr_models.asr_model import ASRModel
from asr_models.sensevoice_model import SenseVoiceASRModel
from asr_models.whisper_model import WhisperASRModel
from asr_datasets.common_voice import CommonVoiceDataset
from asr_datasets.guangzhou_daily_use import GuangzhouDailyUseDataset
from asr_datasets.guangzhou_cabin import GuangzhouCabinDataset
from asr_datasets.asr_dataset import ASRDataset
import torch
import json
import os

datasets: list[ASRDataset] = [
    CommonVoiceDataset(),
    GuangzhouDailyUseDataset(),
    GuangzhouCabinDataset(),
]

device = ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

models: list[ASRModel] = [
    SenseVoiceASRModel(device=device),
    WhisperASRModel(device=device)
]

for dataset in datasets:
    dataset_name = dataset.get_name()
    for model in models:
        model_name = model.get_name()
        results = []
        for batch_audios, batch_sentences in dataset:
            transcriptions = model.generate([audio['array'] for audio in batch_audios])
            for transcription, sentence in zip(transcriptions, batch_sentences):
                results.append({"transcription": transcription["text"], "expected": sentence})

        # Create directory if it doesn't exist
        os.makedirs(f'results/{model_name}', exist_ok=True)
        
        # Save results to a JSON file
        with open(f'results/{model_name}/{dataset_name}.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
