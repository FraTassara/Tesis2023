from kerashypetune import KerasRandomSearchCV
from sklearn.model_selection import KFold
from scipy import stats

def run_search(X, y, n_x, n_y):
    param_grid = {
        'encoder_dim1': stats.randint(5, 15),
        'encoder_dim2': stats.randint(5, 15),
        'decoder_dim': stats.randint(5, 15),
        'activation_1': ['relu'], 
        'activation_2': ['relu'],
        'activation_3': ['relu'],
        'epochs': [100, 120], 
        'batch_size': [16, 32, 64, 128, 256, 512]
    }
    
    cv = KFold(n_splits=10, random_state=33, shuffle=True)
    
    builder = CVAEBuilder(n_x, n_y)
    model_fn = lambda params: builder.build_model(params)[0]
    
    kgs = KerasRandomSearchCV(
        model_fn,
        param_grid,
        cv=cv,
        monitor='loss',
        n_iter=10,
        greater_is_better=False
    )
    
    kgs.search([X, y], X)
    return kgs