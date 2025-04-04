import polars as pl

# Proportion of trajectories to accept
accept_p = 0.10

def summarize_daily(df):
    df = df.with_columns((pl.col('t') // 1).alias('day'))
    df = df.group_by(['day', 'beta_2_value']).agg(pl.sum("Y").alias("Y"))
    df = df.sort('day')
    df = df.rename({"beta_2_value": "proposal"})
    return df

def compare_replicate(data_daily, replicate_data):
    # make sure days are both integers
    data_daily = data_daily.with_columns(pl.col('day').cast(pl.Int64))
    replicate_data = replicate_data.with_columns(pl.col('day').cast(pl.Int64))

    merged_data = data_daily.join(replicate_data, on='day', how='inner')
    merged_data = merged_data.with_columns((merged_data['Y'] - merged_data['Y_right']).alias('difference'))
    score = merged_data['difference'].sum()
    proposal = replicate_data.select(pl.col('proposal')).unique().item()
    return score, proposal

def main():
    results = pl.read_csv("output/results_100_beta.csv")
    fake_data = pl.read_csv("data/fake_outbreak.csv")

    # Initialize a list to store the summary results
    summary_results = []

    # Get unique replicates
    unique_replicates = results['replicate'].unique()

    # Apply the comparison function to each replicate
    for replicate in unique_replicates:
        replicate_data = results.filter(pl.col('replicate') == replicate)
        replicate_data = summarize_daily(replicate_data)
        score, proposal = compare_replicate(fake_data, replicate_data)
        summary_results.append({'replicate': replicate, 'score': score, 'proposal': proposal})

    # Convert the summary results to a DataFrame
    summary_df = pl.DataFrame(summary_results)

    # Calculate the threshold score for the accept_p proportion
    threshold_score = summary_df.select(pl.col('score')).quantile(accept_p).item()

    # Filter the summary results to include only the smallest scores
    accepted_results = summary_df.filter(pl.col('score') <= threshold_score)

    # Write the accepted results to a CSV file
    accepted_results.write_csv("output/abc_summary.csv")

    print("Summary results saved to output/abc_summary.csv")

if __name__ == "__main__":
    main()
