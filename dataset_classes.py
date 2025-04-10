from imports import *
import PHONEMES, config


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, root, phonemes=PHONEMES, context=0, partition="train-clean-100"):
        self.context = context
        self.phonemes = phonemes
        self.subset = config['subset']

        self.freq_masking = tat.FrequencyMasking(freq_mask_param=config['freq_mask_param'])
        self.time_masking = tat.TimeMasking(time_mask_param=config['time_mask_param'])

        self.mfcc_dir = os.path.join(root, partition, "mfcc")
        self.transcript_dir = os.path.join(root, partition, "transcript")

        mfcc_names = sorted(os.listdir(self.mfcc_dir))
        transcript_names = sorted(os.listdir(self.transcript_dir))

        subset_size = int(self.subset * len(mfcc_names))

        mfcc_names = mfcc_names[:subset_size]
        transcript_names = transcript_names[:subset_size]

        assert len(mfcc_names) == len(transcript_names)

        self.mfccs, self.transcripts = [], []

        for i in tqdm(range(len(mfcc_names))):
            # Load a single mfcc
            mfcc_path = os.path.join(self.mfcc_dir, mfcc_names[i])
            mfcc = np.load(mfcc_path)

            # Cepstral Normalization of mfcc along the Time Dimension
            mfccs_normalized = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-5)

            # Convert mfcc to tensor
            mfccs_normalized = torch.tensor(mfccs_normalized, dtype=torch.float32)

            # Load the corresponding transcript
            transcript_path = os.path.join(self.transcript_dir, transcript_names[i])
            transcript = np.load(transcript_path)

            # Remove [SOS] and [EOS] from the transcript
            transcript = transcript[1:-1]

            transcript_indices = [self.phonemes.index(phoneme) for phoneme in transcript]

            # Convert transcript to tensor
            transcript_indices = torch.tensor(transcript_indices, dtype=torch.int64)

            self.mfccs.append(mfccs_normalized)
            self.transcripts.append(transcript_indices)

        self.mfccs = torch.cat(self.mfccs, dim=0)
        print(self.mfccs.shape)

        self.transcripts = torch.cat(self.transcripts, dim=0)

        self.length = len(self.mfccs)

        self.mfccs = nn.functional.pad(self.mfccs, (0, 0, self.context, self.context), mode='constant', value=0)

    def __len__(self):
        return self.length

    def collate_fn(self, batch):
        x, y = zip(*batch)
        x = torch.stack(x, dim=0)

        if np.random.rand() < 0.70:
            x = x.transpose(1, 2)  
            x = self.freq_masking(x)
            x = self.time_masking(x)
            x = x.transpose(1, 2)  

        return x, torch.tensor(y)

    def __getitem__(self, ind):
        frames = self.mfccs[ind : ind + (2 * self.context + 1)]

        phonemes = self.transcripts[ind]

        return frames, phonemes



class AudioTestDataset(torch.utils.data.Dataset):
    def __init__(self, root, context=0, partition="test-clean"):
        self.context = context

        self.subset = 1.0

        self.mfcc_dir = os.path.join(root, partition, 'mfcc')

        self.mfcc_names = sorted(os.listdir(self.mfcc_dir))



        self.mfccs = []

        for i in tqdm(range(len(self.mfcc_names))):
            mfcc_path = os.path.join(self.mfcc_dir, self.mfcc_names[i])
            mfcc = np.load(mfcc_path)

            # Cepstral Normalization of MFCC
            mfcc_normalized = (mfcc - np.mean(mfcc, axis=0)) / np.std(mfcc, axis=0)

            # Convert to tensor and append
            mfcc_normalized = torch.tensor(mfcc_normalized, dtype=torch.float32)
            self.mfccs.append(mfcc_normalized)

        self.mfccs = torch.cat(self.mfccs, dim=0)

        self.length = len(self.mfccs)

        self.mfccs = nn.functional.pad(self.mfccs, (0, 0, self.context, self.context), mode='constant', value=0)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):

        frames = self.mfccs[ind : ind + (2 * self.context) + 1]

        return frames
