from datasets import load_dataset
from datasets import Dataset

class CommonVoiceDataset(Dataset):
    def __init__(self, dataset_path="mozilla-foundation/common_voice_17_0", language="yue", split='test', batch_size=64):
        self.dataset = load_dataset(dataset_path, language, split=split)
        self.batch_size = batch_size
        self.audio_paths = list(self.dataset)
    
    def __iter__(self):
        for i in range(0, len(self.audio_paths), self.batch_size):
            batch = self.audio_paths[i:i + self.batch_size]
            batch_audio_paths = [sample['audio'] for sample in batch]
            batch_sentences = [sample['sentence'] for sample in batch]
            yield batch_audio_paths, batch_sentences

    def get_name(self):
        return "common_voice_17_0"
