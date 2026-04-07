from config import TRUE_PARAMS, TE, sigma
from data_generation import dataset
from plotting import plot_signals


def main():
    sigma_list = sigma
    sigma_val = sigma_list[0]
    clean_signal, noisy_signal = dataset(TE=TE, params=TRUE_PARAMS, sigma=sigma_val)

    print("True Parameters:")
    for key, value in TRUE_PARAMS.items():
        print(f"{key} = {value}")
    print(sigma_val)

    for s in sigma:
        clean_signal, noisy_signal = dataset(TE=TE, params=TRUE_PARAMS, sigma=s)
        plot_signals(
            TE=TE, clean_signal=clean_signal, noisy_signal=noisy_signal, sigma=s
        )


if __name__ == "__main__":
    main()
