{
    'grid_search': {
        '__factory__': 'palladium.fit.with_parallel_backend',
        'estimator': {
            '__factory__': 'sklearn.model_selection.GridSearchCV',
            'estimator': {'__copy__': 'model'},
            'param_grid': {'__copy__': 'grid_search.param_grid'},
            'scoring': {'__copy__': 'scoring'},
        },
        'backend': 'dask',
    },

    '_init_client': {
        '__factory__': 'dask.distributed.Client',
        'address': '127.0.0.1:8786',
    },
}
