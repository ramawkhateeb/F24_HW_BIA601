# ga_module.py
import numpy as np
import pandas as pd
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_POPULATION_SIZE = 50
DEFAULT_NUM_GENERATIONS = 15
DEFAULT_MUTATION_PROBABILITY = 0.01
DEFAULT_CROSSOVER_PROBABILITY = 0.8
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

def load_and_preprocess_dataset(file_path, target_column=None, test_size=0.3, random_seed=RANDOM_SEED):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)

    if target_column is None:
        target_column = df.columns[-1]
    elif target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found --> Available: {list(df.columns)}")

    X_df = df.drop(target_column, axis=1)
    y_series = df[target_column]

    le = LabelEncoder()
    y = le.fit_transform(y_series)

    X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, test_size=test_size, random_state=random_seed)

    numeric_cols = X_train_df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X_train_df.select_dtypes(exclude=np.number).columns.tolist()

    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    column_preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numeric_cols),
            ('cat', cat_pipeline, cat_cols)
        ],
        remainder='passthrough'
    )

    X_train = column_preprocessor.fit_transform(X_train_df)
    X_test = column_preprocessor.transform(X_test_df)
    transformed_feature_names = column_preprocessor.get_feature_names_out()

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": transformed_feature_names,
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols
    }

def initialize_population(population_size, num_features):
    return np.random.randint(2, size=(population_size, num_features))

def apply_feature_mask(feature_matrix, chromosome):
    selected_indices = np.where(chromosome == 1)[0]
    if len(selected_indices) == 0:
        return feature_matrix[:, :0]
    return feature_matrix[:, selected_indices]

def evaluate_chromosome_fitness(chromosome, X_train, X_test, y_train, y_test, alpha=0.9):
    X_train_sel = apply_feature_mask(X_train, chromosome)
    X_test_sel = apply_feature_mask(X_test, chromosome)
    if X_train_sel.shape[1] == 0:
        return 0.0
    model = LogisticRegression(solver='liblinear', multi_class='auto', random_state=RANDOM_SEED)
    model.fit(X_train_sel, y_train)
    preds = model.predict(X_test_sel)
    acc = accuracy_score(y_test, preds)
    num_selected = np.sum(chromosome)
    num_total = X_train.shape[1]
    weighted = (alpha * acc) - ((1 - alpha) * (num_selected / num_total))
    return weighted, acc

def select_via_tournament(population, fitnesses, k=3):
    pop_size = len(population)
    indices = np.random.choice(pop_size, size=k, replace=False)
    winner = indices[np.argmax(fitnesses[indices])]
    return population[winner]

def perform_crossover(parent1, parent2, crossover_prob=DEFAULT_CROSSOVER_PROBABILITY):
    L = len(parent1)
    child1 = parent1.copy()
    child2 = parent2.copy()
    if np.random.rand() < crossover_prob:
        point = np.random.randint(1, L)
        child1[point:] = parent2[point:]
        child2[point:] = parent1[point:]
    return child1, child2

def apply_mutation(chromosome, mut_prob):
    child = chromosome.copy()
    for i in range(len(child)):
        if np.random.rand() < mut_prob:
            child[i] = 1 - child[i]
    return child

def run_genetic_algorithm(X_train, X_test, y_train, y_test,
                          population_size=DEFAULT_POPULATION_SIZE,
                          num_generations=DEFAULT_NUM_GENERATIONS,
                          mutation_probability=DEFAULT_MUTATION_PROBABILITY,
                          crossover_probability=DEFAULT_CROSSOVER_PROBABILITY,
                          alpha=0.9,
                          elitism=True,
                          return_history=False,
                          progress_callback=None):
    num_features = X_train.shape[1]
    population = initialize_population(population_size, num_features)
    best_chromosome = None
    best_fitness = -np.inf

    history = []

    for gen in range(1, num_generations + 1):
        fitnesses = np.zeros(population_size)
        accuracies = np.zeros(population_size)
        for i, chrom in enumerate(population):
            wf, acc = evaluate_chromosome_fitness(chrom, X_train, X_test, y_train, y_test, alpha=alpha)
            fitnesses[i] = wf
            accuracies[i] = acc

        current_best_idx = np.argmax(fitnesses)
        current_best_fit = fitnesses[current_best_idx]
        current_best_chrom = population[current_best_idx].copy()
        current_best_acc = accuracies[current_best_idx]

        if current_best_fit > best_fitness:
            best_fitness = current_best_fit
            best_chromosome = current_best_chrom.copy()
            best_accuracy = current_best_acc

        if progress_callback:
            progress_callback({
                "generation": gen,
                "best_fitness": float(best_fitness),
                "generation_best_fitness": float(current_best_fit),
                "generation_best_accuracy": float(current_best_acc),
                "selected_features_count": int(np.sum(current_best_chrom))
            })

        history.append({
            "generation": gen,
            "generation_best_fitness": float(current_best_fit),
            "generation_best_accuracy": float(current_best_acc),
            "selected_features_count": int(np.sum(current_best_chrom))
        })

        new_pop = []
        if elitism:
            new_pop.append(population[current_best_idx].copy())

        while len(new_pop) < population_size:
            p1 = select_via_tournament(population, fitnesses)
            p2 = select_via_tournament(population, fitnesses)
            c1, c2 = perform_crossover(p1, p2, crossover_prob=crossover_probability)
            c1 = apply_mutation(c1, mutation_probability)
            c2 = apply_mutation(c2, mutation_probability)
            new_pop.extend([c1, c2])

        population = np.array(new_pop[:population_size])

    result = {
        "best_chromosome": best_chromosome,
        "best_fitness": float(best_fitness),
        "best_accuracy": float(best_accuracy) if best_chromosome is not None else 0.0,
        "history": history if return_history else None
    }
    return result
