config = {
    'subset': 1, 
    'context': 40,
    'archetype': 'pyramid', 
    'activations': 'GELU',
    'learning_rate': 0.001,
    'dropout': 0.3,
    'optimizers': 'SGD',
    'scheduler': 'StepLR',
    'epochs': 30,
    'batch_size': 1024,
    'weight_decay': 0.01,
    'weight_initialization': 'kaiming_normal', 
    'augmentations': 'Both', 
    'freq_mask_param': 4,
    'time_mask_param': 8
}