import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
import warnings


RANDOM_STATE = 42
MODEL_PATH = "model.pkl"
ROC_PATH = "roc_curve_logistic_regression.png"


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),
        "SVC": SVC(
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=5),
    }

    results = []
    print("Comparacion de modelos con control basico de sobreajuste:\n")

    for name, model in candidates.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        model.fit(X_train, y_train)
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        test_accuracy = accuracy_score(y_test, model.predict(X_test))
        overfit_gap = train_accuracy - test_accuracy

        results.append(
            {
                "name": name,
                "model": model,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "overfit_gap": overfit_gap,
            }
        )

        print(
            f"{name}: "
            f"cv={cv_scores.mean():.4f} (+/- {cv_scores.std():.4f}) | "
            f"train={train_accuracy:.4f} | "
            f"test={test_accuracy:.4f} | "
            f"gap={overfit_gap:.4f}"
        )

    chosen = next(result for result in results if result["name"] == "LogisticRegression")
    model = chosen["model"]
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    print("\nModelo elegido para despliegue: LogisticRegression")
    print(
        "Se prioriza por interpretabilidad, estabilidad en validacion cruzada "
        "y una brecha train-test contenida, evitando elegir solo por una corrida puntual.\n"
    )
    print(f"Accuracy en test: {chosen['test_accuracy']:.4f}")
    print(f"Accuracy medio CV: {chosen['cv_mean']:.4f}\n")
    print("Classification report:")
    print(classification_report(y_test, predictions, target_names=iris.target_names))

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    roc_auc = roc_auc_score(y_test_bin, probabilities, multi_class="ovr", average="macro")
    print(f"ROC AUC macro (OvR): {roc_auc:.4f}")

    plt.figure(figsize=(8, 6))
    for class_index, class_name in enumerate(iris.target_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, class_index], probabilities[:, class_index])
        class_auc = roc_auc_score(y_test_bin[:, class_index], probabilities[:, class_index])
        plt.plot(fpr, tpr, linewidth=2, label=f"{class_name} (AUC = {class_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Clasificador aleatorio")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC multiclase - LogisticRegression")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_PATH, dpi=150)
    plt.close()

    joblib.dump(model, MODEL_PATH)
    print(f"Modelo guardado correctamente en '{MODEL_PATH}'")
    print(f"Curva ROC guardada en '{ROC_PATH}'")


if __name__ == "__main__":
    main()
