from functools import reduce
import os, glob
from pathlib import Path
import math
from typing import List, Union, Tuple
import sacrebleu
from scipy.stats import norm
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, binom
from types import SimpleNamespace
from random import Random, shuffle
import warnings
import spacy


SPACY_LANG_MODELS = {}
    
def get_lang_model(lang):
    if lang not in SPACY_LANG_MODELS:
        nlp = spacy.blank(lang)
        nlp.add_pipe("sentencizer")
        SPACY_LANG_MODELS[lang] = nlp
    return SPACY_LANG_MODELS[lang]

def split_into_sents(lang_code, text):
    nlp = get_lang_model(lang_code)
    if nlp is None:
        raise ValueError(f"Unsupported language code: {lang_code}")
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def filter_long_comments(df):
    lang_col = 'language' if 'language' in df.columns else 'comment_lang'
    filtered_df = (df[df                                       
        .apply(lambda row: len(split_into_sents(row[lang_col], row['comment'])) <= 3, axis=1)]      
        .reset_index(drop=True)
        )
    return filtered_df


def filter_df(df, dataset):
    lang_col = 'language' if 'language' in df.columns else 'comment_lang'
    filtered_df = (df[(df['comment_lang'] == df['original_lang']) & (df.source == dataset)]      
        .reset_index(drop=True)
        )
    return filtered_df


def compute_accuracy(df, gold_column, pred_column=None, group_by:Union[List, str]=None, baseline=False):
    if baseline:
        mfc = df[gold_column].mode()[0]
        comp = lambda x: (x[gold_column] == mfc).mean()
        columns = [gold_column]
    else:
        if pred_column is None:
            raise ValueError("pred_column must be specified when baseline is False.")
        comp = lambda x: (x[gold_column] == x[pred_column]).mean()
        columns = [gold_column, pred_column] if gold_column != pred_column else [gold_column]

    if group_by:
        return df.groupby(group_by)[columns].apply(comp).reset_index(name='accuracy')

    return comp(df[columns])


def add_accuracy_diff(acc_df, baseline_col: str, diff_mode='baseline'):
    """
    Adds accuracy difference annotations to a DataFrame.
    
    baseline_col: either the first numerical column (in 'previous' mode) or the column 
    against which to compare all other columns (in 'baseline' mode)
    diff_mode: either 'baseline' or 'previous'
    """
    # Validate diff_mode
    if diff_mode not in ['baseline', 'previous']:
        raise ValueError("diff_mode must be either 'baseline' or 'previous'.")

    # Check if the baseline column exists
    if baseline_col not in acc_df.columns:
        raise ValueError(f"Column '{baseline_col}' not found in the DataFrame.")

    # Check if the baseline column is numeric (if using 'baseline' mode)
    if diff_mode == 'baseline' and not pd.api.types.is_numeric_dtype(acc_df[baseline_col]):
        raise ValueError(f"Baseline column '{baseline_col}' must be numeric.")

    # Check if the DataFrame has at least two numerical columns for 'previous' mode
    num_cols = acc_df.select_dtypes(include='number').columns
    if diff_mode == 'previous' and len(num_cols) < 2:
        raise ValueError("At least two numerical columns are required for 'previous' mode.")

    df = acc_df.copy()
    df[num_cols] = (df[num_cols] * 100).round(2)

    # Create a new DataFrame to store formatted values
    formatted_df = df.copy()

    for i, col in enumerate(num_cols):
        if col == baseline_col:
            continue

        ref_col = baseline_col if diff_mode == 'baseline' else num_cols[i - 1]
        formatted_df[col] = df.apply(lambda row: f"{row[col]:.2f}% ({row[col] - row[ref_col]:+.2f})", axis=1)

    # Format the baseline column
    formatted_df[baseline_col] = df[baseline_col].apply(lambda x: f"{x}%")
    
    return formatted_df


def analyse_performance(df, group_col='language'):
    for col in ['comment_pred', 'transformation_pred', 'translation_pred', 'rtt_pred']:
        if col not in df.columns:
            df[col] = float('nan')
            
    # Accuracy
    comment_acc = compute_accuracy(df, 'label', 'comment_pred', group_by=group_col)
    transformation_acc = compute_accuracy(df, 'label', 'transformation_pred', group_by=group_col)
    translation_acc = compute_accuracy(df, 'label', 'translation_pred', group_by=group_col)
    rtt_acc = compute_accuracy(df, 'label', 'rtt_pred', group_by=group_col)

    all_acc = reduce(lambda left, right: pd.merge(left, right, on=group_col), [
        comment_acc.rename(columns={'accuracy': 'comment_pred'}),
        transformation_acc.rename(columns={'accuracy': 'transformation_pred'}),
        translation_acc.rename(columns={'accuracy': 'translation_pred'}),
        rtt_acc.rename(columns={'accuracy': 'rtt_pred'})
    ])

    accuracies_with_diff = add_accuracy_diff(all_acc, 'comment_pred')
    accuracies = accuracies_with_diff.T.rename(columns={0: 'Accuracy'})[1:].reset_index()

    # Error
    comment_err = analyze_errors(df, 'label', 'comment_pred', group_cols=group_col, compute_ci=True)
    transformation_err = analyze_errors(df, 'label', 'transformation_pred', group_cols=group_col, compute_ci=True)
    translation_err = analyze_errors(df, 'label', 'translation_pred', group_cols=group_col, compute_ci=True)
    rtt_err = analyze_errors(df, 'label', 'rtt_pred', group_cols=group_col, compute_ci=True)

    errors = pd.concat(
        [comment_err, transformation_err, translation_err, rtt_err]
        ).drop(columns=[group_col]).reset_index(drop=True)

    return pd.concat([accuracies, errors], axis=1)


def plot_metrics(model_dir, multi_eval:bool=False, save:str=''):

    def _annotate_values(ax, x, y, label, color):
        # Annotate initial value
        ax.annotate(f'{y.iloc[0]:.2f}', (x.iloc[0], y.iloc[0]), textcoords="offset points", xytext=(-10,10), ha='center', color=color)
        # Annotate final value
        ax.annotate(f'{y.iloc[-1]:.2f}', (x.iloc[-1], y.iloc[-1]), textcoords="offset points", xytext=(-10,-10), ha='center', color=color)
        # Annotate lowest (for losses) or highest (for accuracies), if it is not the same as the initial or final values
        best_idx = y.idxmin() if 'loss' in label else y.idxmax()
        if (best_idx != y.index[-1]) and (best_idx != y.index[0]):  
            ax.annotate(f'{y.loc[best_idx]:.2f}', (x.loc[best_idx], y.loc[best_idx]), 
                        textcoords="offset points", xytext=(10,-10), ha='center', color=color)
    
    def _plot_single_eval(log_history):
        plt.style.use('ggplot')
        metric_cols = log_history[['loss', 'eval_loss', 'eval_accuracy']].columns
        colors = ['maroon', 'red', 'green']

        for column, color in zip(metric_cols, colors):
            x = log_history[~log_history[column].isna()]['epoch']
            y = log_history[column].dropna()
            plt.plot(x, y, label=column, color=color)
            _annotate_values(plt, x, y, column, color=color)

        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(save, format='pdf', bbox_inches='tight')
            print(f"Figure saved to {save}")
        plt.show()

    def _plot_multi_eval(log_history):
        plt.style.use('ggplot')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        losses = [col for col in log_history.columns if col.endswith('loss')]
        accuracies = [col for col in log_history.columns if col.endswith('accuracy')]

        datasets = [col.split('_valid.csv_loss')[0].replace('loss', 'train') for col in losses]
        color_dict = {dataset: plt.cm.tab10(i) for i, dataset in enumerate(datasets)}

        # Plot accuracies
        for acc_col in accuracies:
            dataset = acc_col.split('_valid.csv_accuracy')[0]
            x_acc = log_history[~log_history[acc_col].isna()]['epoch']
            y_acc = log_history[acc_col].dropna()
            ax1.plot(x_acc, y_acc, label=dataset, color=color_dict[dataset])
            _annotate_values(ax1, x_acc, y_acc, acc_col, color=color_dict[dataset])
        
        # Plot losses
        for loss_col in losses:
            dataset = loss_col.split('_valid.csv_loss')[0].replace('loss', 'train')
            x_loss = log_history[~log_history[loss_col].isna()]['epoch']
            y_loss = log_history[loss_col].dropna()
            ax2.plot(x_loss, y_loss, label=dataset, color=color_dict[dataset])
            _annotate_values(ax2, x_loss, y_loss, loss_col, color=color_dict[dataset])

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        # ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        if save:
            plt.savefig(save, format='pdf', bbox_inches='tight')
            print(f"Figure saved to {save}") 
        plt.show()
    
    # Sort checkpoint files by modification time
    ckpt_files = sorted(glob.glob(os.path.join(model_dir, 'checkpoint-*')), key=os.path.getmtime)

    if not ckpt_files:
        print(f"No checkpoint files in {model_dir}")
        return  

    trainer_state_path = os.path.join(ckpt_files[-1], 'trainer_state.json')

    if not os.path.exists(trainer_state_path):
        print(f"trainer_state.json not found in {ckpt_files[-1]}")
        return

    with open(trainer_state_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    log_history = (pd.DataFrame(data['log_history'])
                .groupby('epoch', as_index=False)
                .first()
                )

    if multi_eval:
        _plot_multi_eval(log_history)
    else:
        _plot_single_eval(log_history)


def wald_ci(p, n, confidence=0.95):
    """
    Computes a confidence interval using the Gaussian (Wald) method.

    Args:
        p (float): Proportion of errors (or any rate being measured).
        n (int): Sample size.
        confidence (float): Confidence level (default is 0.95 for 95% CI).

    Returns:
        str: Confidence interval as a formatted string "[lower, upper]".
    """
    if n == 0:
        return "[0, 0]"
    
    z = norm.ppf(1 - (1 - confidence) / 2)  # Compute z-score dynamically
    se = np.sqrt(p * (1 - p) / n)  # Standard error
    lower = max(0, p - z * se)  # Clamp at 0 to avoid negative proportions
    upper = min(1, p + z * se)  # Clamp at 1 to avoid >100% values
    
    return f"[{round(lower * 100, 2)}, {round(upper * 100, 2)}]"


def analyze_errors(df, gold_column, pred_column, group_cols, confidence=0.95, compute_ci=False):
    """
    Analyzes prediction errors by language, computing error rates and (optional) confidence intervals.
    """
    def compute_stats(group):
        n = len(group)
        err_rate = np.mean(group[gold_column] != group[pred_column])
        fp_rate = np.mean((group[gold_column] == 'AGAINST') & (group[pred_column] == 'FAVOR'))
        fn_rate = np.mean((group[gold_column] == 'FAVOR') & (group[pred_column] == 'AGAINST'))

        result = {
            'ErrRate (%)': round(err_rate * 100, 2),
            'FP (%)': round(fp_rate * 100, 2),
            'FN (%)': round(fn_rate * 100, 2)
        }

        if compute_ci:
            result.update({
                'ErrRate_CI': wald_ci(err_rate, n, confidence),
                'FP_CI': wald_ci(fp_rate, n, confidence),
                'FN_CI': wald_ci(fn_rate, n, confidence)
            })
        
        return pd.Series(result)
    
    return df.groupby(group_cols)[[gold_column, pred_column]].apply(compute_stats).reset_index()


def balance_df(df, columns: List, return_filtered: bool = False, rs=42):
    # Minimum count of rows for any combination of values in the specified columns
    min_count = df.groupby(columns).size().min()
    
    # Sample min_count rows for each column combination
    balanced_df = (
        df.groupby(columns)
          .apply(lambda group: group.sample(min_count, random_state=rs))
          .reset_index(drop=True)
    )
    
    # Shuffle the rows in the balanced DataFrame
    balanced_df = balanced_df.sample(frac=1, random_state=rs).reset_index(drop=True)
    
    if return_filtered:
        # Concatenate and find unique rows to identify filtered-out samples
        filtered_out = pd.concat([df, balanced_df]).drop_duplicates(keep=False)
        return balanced_df, filtered_out
    
    return balanced_df


def compute_bleu(src_sentences: List[str], rtt_sentences: List[str]) -> Tuple[float, List[float]]:
    """
    Compute corpus and sentence BLEU between originals and RTTs.
    Returns (corpus_bleu, sentence_bleu_list).
    """
    if len(src_sentences) != len(rtt_sentences):
        raise ValueError("Input lists must have identical length")

    # corpus BLEU
    # corpus_bleu = round(sacrebleu.corpus_bleu(rtt_sentences, [src_sentences]).score, 2)   # float
    corpus_bleu = sacrebleu.corpus_bleu(rtt_sentences, [src_sentences]).score   # float

    # sentence BLEU
    sentence_bleu = [
        # round(sacrebleu.sentence_bleu(hyp, [ref]).score, 2)
        sacrebleu.sentence_bleu(hyp, [ref]).score
        for hyp, ref in zip(rtt_sentences, src_sentences)
    ]

    return corpus_bleu, sentence_bleu   # rounded to 2 decimal places


def mcnemar_test(B, A, Y, alpha=0.05):
    """
    Performs McNemar's test with fallback to one-tailed exact binomial test for small samples.
    
    Parameters:
        B (List[str]): Predictions before intervention
        A (List[str]): Predictions after intervention
        Y (List[str]): Gold labels
        alpha (float): Significance level (default 0.05)
    
    Returns:
        p_value (float): Computed p-value under the null hypothesis that the marginal distributions are equal
        test_type (str): 'chi2' or 'exact-binomial'
        significant (bool): Whether the result is significant at given alpha
    """
    b = sum(b == y and a != y for b, a, y in zip(B, A, Y))  # Count of correct → incorrect flips
    c = sum(b != y and a == y for b, a, y in zip(B, A, Y))  # Count of incorrect → correct flips
    n = b + c

    if n >= 25:
        # Use chi-squared approximation
        chi2_stat = (b - c)**2 / n
        p_value = 1 - chi2.cdf(chi2_stat, df=1)  # two-tailed by default
        test_type = 'chi2'
    else:
        # Use one-tailed exact binomial test
        p_value = sum(binom.pmf(k, n, 0.5) for k in range(b, n + 1))
        test_type = 'exact-binomial'

    significant = p_value < alpha
    return p_value, test_type, significant


def z_test(population1, population2, label, alpha=0.05, two_tailed=False, continuity=True):

    """ Test 5 in 100 statistical Tests """
    def z_test_two_proportions(p1, n1, p2, n2, alpha=0.05, two_tailed=False):
        P = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = math.sqrt(P * (1 - P) * (1/n1 + 1/n2)) # standard error
        z = (p1 - p2) / se

        if two_tailed:
            critical = norm.ppf(1 - alpha / 2)
            reject_h0 = abs(z) > critical 
        else:   # right-sided one-tailed test
            critical = norm.ppf(1 - alpha)
            reject_h0 = z > critical # if z < -critical -> the translations resulted in better predictions than the originals

        return z, critical, reject_h0

    population1_consistency = population1.value_counts().get(label, 0)
    population2_consistency = population2.value_counts().get(label, 0)

    n1 = len(population1)
    n2 = len(population2)

    p1 = population1_consistency / n1
    p2 = population2_consistency / n2
    
    z, critical, reject_h0 = z_test_two_proportions(p1, n1, p2, n2, alpha)
    
    return round(float(z), 2), round(float(critical), 2), bool(reject_h0)


def construct_bias_variants(biased_df, neutral_df, intervals=10, seed=42):
    """
    Generates datasets with varying levels of bias by combining biased and neutral samples.

    Parameters:
    - biased_df (DataFrame): Contains labeled biased examples ('FAVOR' or 'AGAINST').
    - neutral_df (DataFrame): Contains neutral examples, some overlapping with biased_df.

    Returns:
    - pos_sets (dict): Datasets with 0–50% bias injected for 'FAVOR' samples.
    - neg_sets (dict): Datasets with 0–50% bias injected for 'AGAINST' samples.

    Notes:
    - Bias is injected by replacing neutral samples with their biased counterparts.
    - Random neutral samples are added to balance dataset sizes.
    """
    pos_biased = biased_df[biased_df.label == 'FAVOR'].reset_index(drop=True)
    neg_biased = biased_df[biased_df.label == 'AGAINST'].reset_index(drop=True)

    # print(f"pos_biased: {len(pos_biased)}")
    # print(f"neg_biased: {len(neg_biased)}")

    # neutral translations of the items in the biased sample
    neutral_overlap = neutral_df[neutral_df.id.isin(biased_df.id)]
    if len(neutral_overlap) != len(biased_df):
        print(f"{len(biased_df) - len(neutral_overlap)} items from the biased set are missing from the neutral set.")
    neu_pos_overlap = neutral_overlap[neutral_overlap.label == 'FAVOR']
    neu_neg_overlap = neutral_overlap[neutral_overlap.label == 'AGAINST']
    # same number of random neutral translations (no overlap)
    neu_pos_random = neutral_df[(~neutral_df.id.isin(biased_df.id)) & (neutral_df.label == 'FAVOR')].sample(len(pos_biased), random_state=42)
    neu_neg_random = neutral_df[(~neutral_df.id.isin(biased_df.id)) & (neutral_df.label == 'AGAINST')].sample(len(neg_biased), random_state=42)

    pos_neutral = pd.concat([neu_pos_overlap, neu_pos_random]).reset_index(drop=True)
    neg_neutral = pd.concat([neu_neg_overlap, neu_neg_random]).reset_index(drop=True)

    # print(f"pos_neutral: {len(pos_neutral)}")
    # print(f"neg_neutral: {len(neg_neutral)}")

    def inject_bias(biased_set, neutral_set):
        sets = {}
        # Shuffle biased IDs reproducibly if seed is provided
        biased_ids = biased_set['id'].tolist()
        if seed is not None:
            rng = Random(seed)
            rng.shuffle(biased_ids)
        else:
            shuffle(biased_ids)
        neutral_base = neutral_set.set_index('id')

        for pct in range(0, len(biased_ids)+intervals, intervals):
            i = round(len(neutral_base) * pct/100)
            # print(f"i = {i}")
            current_biased_ids = biased_ids[:i]
            biased_cumulative = biased_set[biased_set['id'].isin(current_biased_ids)].set_index('id')

            neutral_copy = neutral_base.copy()
            neutral_copy['transformation'] = neutral_copy['transformation'].astype('object')
            neutral_copy['transformation_pred'] = neutral_copy['transformation_pred'].astype('object')
            # Replace matching rows in neutral set with biased versions
            neutral_copy.update(biased_cumulative)

            mixed_set = neutral_copy.reset_index()
            sets[f'bias_{pct}%'] = mixed_set

        return sets

    pos_sets = inject_bias(pos_biased, pos_neutral)
    neg_sets = inject_bias(neg_biased, neg_neutral)

    return pos_sets, neg_sets


def convert_to_cometkiwi_input(x, src_col, mt_col, save_dir, filename=''):
    parent_path = Path(save_dir)
    parent_path.mkdir(parents=True, exist_ok=True)

    if isinstance(x, pd.DataFrame):
        input_dict = {'ck_input': x}
    elif isinstance(x, dict):
        input_dict = x  # x must be {str: df}
    else:
        print("Incorrect input type.")
        return

    for key, value in input_dict.items():
        name = filename if filename else f'{key}.json'
        file_path = parent_path / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data = value[[src_col, mt_col]].rename(columns={src_col: 'src', mt_col: 'mt'}).to_dict(orient="records")
        with open(file_path, 'w', encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Data written to {file_path}")


def mcnemar_sensitivity(biased_sets, rtt=False, alpha=0.05):
    scores = {}
    for name, dataset in biased_sets.items():
        B = dataset['comment_pred']                                        # (B)efore
        A = dataset['rtt_pred'] if rtt else dataset['translation_pred']    # (A)fter
        Y = dataset['label']                                               # Labels

        p_value, test_type, significant = mcnemar_test(B, A, Y, alpha=alpha)
        if significant == True:
            pct = name.split('_')[-1].strip('%')
            return int(pct), p_value


def z_score_sensitivity(df, biased_sets, label, language='de', convert_to_p=False, rtt=False, alpha=0.05):
    population1 = df[(df['language'] == language) & (df['label'] == label)]['comment_pred']
    scores = {}
    for name, dataset in biased_sets.items():                                    
        population2 = dataset['rtt_pred'] if rtt else dataset['translation_pred']                                           

        z, critical, significant = z_test(
                population1=population1,
                population2=population2,
                label=label,
                alpha=alpha,
                return_results=True
            )
        
        if convert_to_p:
            score = norm.sf(z)
        else:
            score = z

        if significant == True:
            pct = name.split('_')[-1].strip('%')
            return int(pct), score
    
    return np.nan, score


def monte_carlo_simulation(
    biased,
    neutral,
    intervals,
    sensitivity_fn,
    df=None,
    language=None,
    rtt=False,
    alpha=0.05,
    num_runs=10,
    convert_to_p=False
):
    """Generic Monte Carlo evaluation for different sensitivity tests."""
    
    def format_result(avg_pct, avg_stat, statistic='p'):
        stat_str = f"{avg_stat:.2f}" if statistic == 'Z' else f"{avg_stat:.3f}"
        if np.isnan(avg_pct):
            return f"N/A ({statistic}={stat_str})"
        return f"{int(avg_pct)}% ({statistic}={stat_str})"

    
    all_pos, all_neg = [], []

    for seed in range(num_runs):
        biased_pos_sets, biased_neg_sets = construct_bias_variants(
            biased, neutral, seed=seed, intervals=intervals
        )

        if sensitivity_fn.__name__.startswith("z_score"):
            result_pos = sensitivity_fn(
                df, biased_pos_sets, label="FAVOR",
                language=language, convert_to_p=convert_to_p,
                rtt=rtt, alpha=alpha
            )
            result_neg = sensitivity_fn(
                df, biased_neg_sets, label="AGAINST",
                language=language, convert_to_p=convert_to_p,
                rtt=rtt, alpha=alpha
            )
        else:  # assume McNemar
            result_pos = sensitivity_fn(biased_pos_sets, rtt=rtt, alpha=alpha)
            result_neg = sensitivity_fn(biased_neg_sets, rtt=rtt, alpha=alpha)

        all_pos.append(result_pos)
        all_neg.append(result_neg)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        avg_pct_pos, avg_stat_pos = np.nanmean(all_pos, axis=0)
        avg_pct_neg, avg_stat_neg = np.nanmean(all_neg, axis=0)

    # Output formatting
    statistic = "p" if sensitivity_fn.__name__.startswith("mcnemar") or convert_to_p else "Z"
    return (
        format_result(avg_pct_pos, avg_stat_pos, statistic),
        format_result(avg_pct_neg, avg_stat_neg, statistic)
    )