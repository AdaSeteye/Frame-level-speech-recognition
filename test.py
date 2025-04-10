from imports import device, torch, tqdm
import PHONEMES
from train import model
from data_loader import test_loader


def test(model, test_loader):
    model.eval() 

    test_predictions = []

    with torch.no_grad(): # TODO

        for i, mfccs in enumerate(tqdm(test_loader)):

            mfccs   = mfccs.to(device)

            logits  = model(mfccs)

            predicted_phonemes = torch.argmax(logits, dim=1)

            
            predicted_phonemes = [PHONEMES[i] for i in predicted_phonemes.cpu().numpy()]
            for elt in predicted_phonemes:
                test_predictions.append(elt)

    return test_predictions



predictions = test(model, test_loader)