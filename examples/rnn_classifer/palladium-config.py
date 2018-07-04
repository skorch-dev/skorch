{
    'dataset_loader_train': {
        '__factory__': 'model.DatasetLoader',
        'path': 'aclImdb/train/',
    },

    'dataset_loader_test': {
        '__factory__': 'model.DatasetLoader',
        'path': 'aclImdb/test/',
    },

    'model': {
        '__factory__': 'model.create_pipeline',
        'use_cuda': True,
    },

    'model_persister': {
        '__factory__': 'palladium.persistence.File',
        'path': 'rnn-model-{version}',
    },

    'grid_search': {
        'param_grid': {
            'to_idx__stop_words': ['english', None],
            'to_idx__lowercase': [False, True],
            'to_idx__ngram_range': [(1, 1), (2, 2)],
            'net__module__embedding_dim': [32, 64, 128, 256],
            'net__module__rec_layer_type': ['gru', 'lstm'],
            'net__module__num_units': [32, 64, 128, 256],
            'net__module__num_layers': [1, 2, 3],
            'net__module__dropout': [0, 0.25, 0.5, 0.75],
            'net__lr': [0.003, 0.01, 0.03],
            'net__max_epochs': [5, 10],
        },
    },

    'scoring': 'accuracy',

    'predict_service': {
        '__factory__': 'palladium.server.PredictService',
        'mapping': [
            ('text', 'str'),
        ],
        'predict_proba': True,
        'unwrap_sample': True,
    },

}
