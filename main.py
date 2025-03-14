import argparse
import logging

from data import load_twins_data, prepare_train_test_split
from train_test import train_and_evaluate

def main():
    parser = argparse.ArgumentParser(
        description="CEVAE with Z and other model comparisons using combined XZ features."
    )
    parser.add_argument("--num-data", default=23968, type=int)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--cv-folds", default=5, type=int)
    parser.add_argument("--latent-dim", default=30, type=int)
    parser.add_argument("--feature-dim", default=44, type=int) 
    parser.add_argument("--hidden-dim", default=300, type=int)
    parser.add_argument("--num-layers", default=2, type=int)
    parser.add_argument("-n", "--num-epochs", default=40, type=int)
    parser.add_argument("-b", "--batch-size", default=512, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-3, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.95, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--seed", default=66, type=int)
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    # ---------------- NEW ARGUMENT ----------------
    parser.add_argument("--only-cevae", action="store_true", default=True, help="If set, only train/evaluate CEVAE, skipping other models.")

    args = parser.parse_args()

    logging.getLogger("pyro").setLevel(logging.DEBUG)
    if logging.getLogger("pyro").handlers:
        logging.getLogger("pyro").handlers[0].setLevel(logging.DEBUG)

    # data loading
    X, t, y, Z = load_twins_data()

    # perform train/test dataset splits
    (X_train, t_train, y_train, Z_train,
     X_test,  t_test,  y_test,  Z_test,
     true_ite_train, true_ite_test,
     XZ_train, t_train_np, y_train_np,
     train_twin0, train_twin1,
     test_twin0, test_twin1) = prepare_train_test_split(
        X, t, y, Z,
        num_data=args.num_data,
        test_size=args.test_size,
        seed=args.seed
    )

    # run model training and evaluation
    train_and_evaluate(
        args,
        X, t, y, Z,
        X_train, t_train, y_train, Z_train,
        X_test,  t_test,  y_test,  Z_test,
        true_ite_train, true_ite_test,
        XZ_train, t_train_np, y_train_np,
        train_twin0, train_twin1,
        test_twin0, test_twin1
    )


if __name__ == "__main__":
    main()
