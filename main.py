from data.loader import DataLoader
from training.tuner import run_search
from models.cvae import CVAEBuilder

if __name__ == "__main__":
    # Cargar datos
    loader = DataLoader('lc_database.csv')
    X, y, scaler, feature_names = loader.load_and_preprocess()
    
    # Búsqueda de hiperparámetros
    n_x = X.shape[1]
    n_y = y.shape[1]
    kgs = run_search(X, y, n_x, n_y)
    
    # Mejor modelo
    best_model = kgs.best_model
    best_params = kgs.best_params
    
    # Generar datos sintéticos (ejemplo)
    builder = CVAEBuilder(n_x, n_y)
    _, mu, l_sigma = builder.build_model(best_params)
    # ... (agregar lógica de generación)