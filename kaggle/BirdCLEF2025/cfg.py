class CFG:
 
    test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
    submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    model_path = '/kaggle/input/birdclef-2025-efficientnet-b0'  
    
    # Audio parameters
    FS = 32000  
    WINDOW_SIZE = 5  
    
    # Mel spectrogram parameters
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 149
    FMIN = 50
    FMAX = 14000
    TARGET_SHAPE = (256, 256)
    
    model_name = 'efficientnet_b0'
    in_channels = 1
    device = 'cpu'  
    
    # Inference parameters
    batch_size = 16
    use_tta = False  
    tta_count = 3   
    threshold = 0.5
    
    use_specific_folds = False  # If False, use all found models
    folds = [0, 1]  # Used only if use_specific_folds is True
    
    debug = False
    debug_count = 3