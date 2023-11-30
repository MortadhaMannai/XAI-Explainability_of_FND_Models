import matplotlib.pyplot as plt
import numpy as np
import shap

# for interactive plots
shap.initjs()


def barplot_token_shap_values(tokens: list, shap_values: list, label: int, save_as=None):
    """plot the tokens against their shap values as a horizontal barplot
    Parameters
    ----------
    tokens: list(str)
        list of tokens to be visualized
    shap_values: list(int),
        list of shap values of the tokens
    label: int,
        label of the current sample whose tokens and shap values will be visualized.
    save_as: str,
        if set to none does not save, if set to any str value, saves barplot and forceplot under
        plot_images/<save_as>.pdf
    """
    plt.figure(figsize=(16, 6))
    if label == 0:
        label_color = 'red'
    else:
        label_color = 'green'
    # put each token in quotation marks to spot the space character
    tokens_visible = []
    for token in tokens:
        tokens_visible.append(f'\"{token}\"')
    # plt.barh(tokens_visible, shap_values, color=label_color)
    plt.bar(tokens_visible, shap_values, color=label_color)
    # Add Plot Title
    plt.title(f'First {len(tokens)} important tokens', loc='left', )

    if save_as is not None:
        plt.savefig(f'plot_images/{save_as}_barplot.pdf', bbox_inches='tight')
    # Show Plot
    plt.show()


def get_most_important_n_tokens(shap_values_dict, label, n=10, verbose=False):
    """collect n most important tokens from the shap_values_dict
    Parameters
    ----------
    shap_values_dict: dict
        the shap values of the sample
    label: int
        the actual label of the sample
    n: int
        how many tokens to get. The default is 10
    verbose: bool,
        whether to print out additional information. The default is False
    """
    # we keep this sum to understand how much of the overall is made up by the important ones.
    tokenized_input = shap_values_dict.data
    base_values = shap_values_dict.base_values
    if verbose:
        print(f'Base values are: {base_values[0]} and  {base_values[1]}')

    shap_values = shap_values_dict.values[:, label]

    print(f'Max shap value is {np.max(shap_values)}')

    first_n_important_indexes = np.argsort(shap_values)[-n:]
    first_n_important_shap_values = shap_values[first_n_important_indexes]
    if verbose:
        for idx in first_n_important_indexes:
            print(f'\n--> "{tokenized_input[idx]}" with shap value: {shap_values[idx]}')

    n_cumulative_importance = np.sum(first_n_important_shap_values, axis=0)
    # we take only the positive values for comparison. This value is very close to base_values[0]
    positive_shap_values_sum = np.sum(shap_values[np.where(shap_values > 0)], axis=0)
    if verbose:
        print(f'\nAll {n} tokens are {n * 100 / shap_values.shape[0]}% of all tokens.\nTheir cumulative importance is '
              f'{n_cumulative_importance}.\nThis value shows that '
              f'{n_cumulative_importance * 100 / positive_shap_values_sum}% of cumulative importance of all positive '
              f'contributing tokens ({positive_shap_values_sum}) are these {n} tokens.\n')
        print("###################################################################")

    return tokenized_input[first_n_important_indexes], first_n_important_shap_values


def barplot_first_n_largest_shap_values(shap_values, label, n=10, save_as=None):
    """wrapper method for plotting the most important n tokens
    Parameters
    ----------
    shap_values: dict
        the shap values of the sample
    label: int
        the actual label of the sample
    n: int
        how many tokens to barplot. The default is 10
    save_as: str,
        if set to none does not save, if set to any str value, saves barplot and forceplot under
        plot_images/<save_as>.pdf
    """
    tokens, s_values = get_most_important_n_tokens(shap_values, label=label, n=n)
    barplot_token_shap_values(tokens, s_values, label, save_as=save_as)
